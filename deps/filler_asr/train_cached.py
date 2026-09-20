"""
Training driver for the filler_asr SA run on a PRECOMPUTED Arrow cache (built once
by prepare_data_local.py from LOCAL wavs). Same recipe as train_local.py, but the
data is read ready-made -- feature-extracted input_values + framewise <fill>
labels -- and only PADDED at batch time via the original DataCollatorForFillerASR.
No per-step audio decode. This mirrors the original dataset.get_dataset() +
DataCollatorForFillerASR path that the reference box used.

Recipe (from config, unchanged): HuBERT-xlarge encoder + feature extractor FROZEN,
only the 2 self-attention layers + lm_head train, LR = linear warmup -> decay
(lr_schedule_type "linear", classifier_only_train_ratio 0 -> no freezing curriculum).

Run (4 GPUs) -- prepare_data_local.py must have completed first:
  bash run_sa_cached.sh
"""
import os
import json
import argparse
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    HubertConfig,
)
from model import FillerASRTrainer
from model_sa import FillerHubertSAModel
from collator import DataCollatorForFillerASR
from utils import get_compute_metrics_fn
from data_cached import build_cached_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config_xlarge_sa_cached.json")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    out_dir = config.get("output_dir", "experiments")
    exp_name = config.get("exp_name", "Hubert_SA_cached")
    os.environ["WANDB_PROJECT"] = config.get("wandb_project", "filler_asr")
    num_proc = config.get("num_processes", 8)
    model_checkpoint = config["model_checkpoint"]
    cache_dir = config["cache_dir"]
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if not os.path.exists(os.path.join(cache_dir, "PREPARED")):
        raise FileNotFoundError(
            f"Precomputed cache not found at {cache_dir}. Run "
            f"`python prepare_data_local.py --config_path {args.config_path}` first."
        )

    max_steps = args.max_steps if args.max_steps != -1 else config.get("max_steps", -1)
    do_eval = config.get("do_eval", True)

    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 32),
        eval_strategy="steps" if do_eval else "no",
        num_train_epochs=config.get("num_train_epochs", 100),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=True,  # frozen hubert leaves many params grad-less
        save_steps=config.get("save_steps", 1000),
        eval_steps=config.get("eval_steps", 1000),
        logging_steps=config.get("logging_steps", 50),
        learning_rate=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 0.005),
        warmup_steps=config.get("warmup_steps", 0),
        save_total_limit=config.get("save_total_limit", 1),
        max_steps=max_steps,
        report_to="wandb",
        run_name=exp_name,
        dataloader_num_workers=num_proc,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        remove_unused_columns=False,
    )

    # tokenizer + feature extractor were saved next to the cache by prepare
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cache_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(cache_dir)

    # Load precomputed datasets (cache hit; main process leads, others follow)
    print("Loading precomputed datasets ...")
    with training_args.main_process_first(desc="load cached datasets"):
        dataset = build_cached_datasets(config, feature_extractor, tokenizer)
    dev_limit = config.get("dev_limit")
    if dev_limit:
        dataset["test"] = dataset["test"].select(range(min(dev_limit, len(dataset["test"]))))
    print(f"train={len(dataset['train'])}  dev={len(dataset['test'])}")

    # Model: frozen CTC-finetuned xlarge encoder + 2 SA layers + fresh lm_head
    print(f"Loading model from {model_checkpoint} ...")
    hubert_config = HubertConfig.from_pretrained(model_checkpoint)
    hubert_config.vocab_size = len(tokenizer)
    hubert_config.pad_token_id = tokenizer.pad_token_id
    hubert_config.bos_token_id = tokenizer.bos_token_id
    hubert_config.eos_token_id = tokenizer.eos_token_id
    hubert_config.num_sa_layers = config.get("num_sa_layers", 4)
    hubert_config.sa_nhead = config.get("sa_nhead", hubert_config.num_attention_heads)
    hubert_config.sa_dim_feedforward = config.get("sa_dim_feedforward", hubert_config.intermediate_size)
    hubert_config.sa_dropout = config.get("sa_dropout", hubert_config.final_dropout)
    print(f"FillerHubertSAModel: {hubert_config.num_sa_layers} SA layers "
          f"(nhead={hubert_config.sa_nhead}, ff={hubert_config.sa_dim_feedforward})")

    model = FillerHubertSAModel.from_pretrained(
        model_checkpoint, config=hubert_config, ignore_mismatched_sizes=True,
    )
    freeze_fe = config.get("freeze_feature_extractor", True)
    if freeze_fe:
        model.freeze_feature_extractor()

    if training_args.local_rank <= 0:
        model.hubert.save_pretrained(os.path.join(exp_dir, "model"))
        feature_extractor.save_pretrained(exp_dir)
        tokenizer.save_pretrained(exp_dir)

    data_collator = DataCollatorForFillerASR(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        frames_per_sec=config.get("frames_per_sec", 25),
    )
    compute_metrics = get_compute_metrics_fn(tokenizer)

    trainer = FillerASRTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        lr_schedule_type=config.get("lr_schedule_type", "linear"),
        lr_schedule_ratios=config.get("lr_schedule_ratios", [0.1, 0.4, 0.5]),
        lr_exp_final_ratio=config.get("lr_exp_final_ratio", 0.01),
        classifier_only_train_ratio=config.get("classifier_only_train_ratio", 0.0),
        freeze_feature_extractor=freeze_fe,
        fill_weight=config.get("fill_weight", 1.0),
        fill_token_id=tokenizer.convert_tokens_to_ids("<fill>"),
    )

    resume = None
    ckpts = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")] if os.path.isdir(exp_dir) else []
    if ckpts:
        resume = max([os.path.join(exp_dir, d) for d in ckpts], key=os.path.getmtime)
        print(f"Resuming from {resume}")

    trainer.train(resume_from_checkpoint=resume)
    print("TRAIN_CACHED_DONE")


if __name__ == "__main__":
    main()
