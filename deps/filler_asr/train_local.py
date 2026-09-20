"""
Training driver for filler_asr on LOCAL LibriSpeech with hubert-xlarge-ls960-ft.
Single-GPU or multi-GPU (DDP via torchrun).

Reuses the real pipeline unchanged:
  * FillerHubertModel + FillerASRTrainer (triphase LR, freezing curriculum,
    framewise cross-entropy)  from model.py
  * get_compute_metrics_fn  from utils.py
Only the data ingestion is swapped to data_local (local jsonl + lazy audio).

Run (2 GPUs, e.g. CUDA 4 and 6):
  CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 --master_port=29503 \
      train_local.py --config_path config_xlarge_full.json
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
from model import FillerASRTrainer, FillerHubertModel
from utils import get_compute_metrics_fn
from data_local import build_datasets, write_vocab, DataCollatorLazyFillerASR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config_xlarge_full.json")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    out_dir = config.get("output_dir", "experiments")
    exp_name = config.get("exp_name", "Hubert_finetuning")
    os.environ["WANDB_PROJECT"] = config.get("wandb_project", "filler_asr")
    num_proc = config.get("num_processes", 4)
    model_checkpoint = config["model_checkpoint"]
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    max_steps = args.max_steps if args.max_steps != -1 else config.get("max_steps", -1)
    do_eval = config.get("do_eval", True)

    # 1. TrainingArguments first (so we can use distributed state / main_process_first)
    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 32),
        eval_strategy="steps" if do_eval else "no",
        num_train_epochs=config.get("num_train_epochs", 100),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        ddp_find_unused_parameters=True,  # freezing curriculum leaves params grad-less in phase A
        save_steps=config.get("save_steps", 1000),
        eval_steps=config.get("eval_steps", 1000),
        logging_steps=config.get("logging_steps", 50),
        learning_rate=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 0.005),
        warmup_steps=config.get("warmup_steps", 0),
        save_total_limit=config.get("save_total_limit", 2),
        max_steps=max_steps,
        report_to="wandb",
        run_name=exp_name,
        dataloader_num_workers=num_proc,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        remove_unused_columns=False,
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    # 2. Vocab on main process only; other ranks wait then read it
    vocab_file = os.path.join(exp_dir, "vocab.json")
    with training_args.main_process_first(desc="vocab extraction"):
        if training_args.local_rank <= 0:
            print("Extracting vocabulary from local train manifest ...")
            write_vocab(config["train_jsonl"], out_dir, exp_name)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file, unk_token="<unk>", pad_token="<pad>",
        word_delimiter_token="|", bos_token="<s>", eos_token="</s>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<fill>"]})

    # 3. Build datasets (all ranks; tiny in-memory Arrow manifest, lazy audio)
    print("Building local datasets ...")
    dataset = build_datasets(config["train_jsonl"], config["dev_jsonl"],
                             dev_limit=config.get("dev_limit"))
    print(f"train={len(dataset['train'])}  dev={len(dataset['test'])}")

    # 4. Model: pretrained (CTC-finetuned) xlarge encoder + fresh head
    print(f"Loading model from {model_checkpoint} ...")
    hubert_config = HubertConfig.from_pretrained(model_checkpoint)
    hubert_config.vocab_size = len(tokenizer)
    hubert_config.pad_token_id = tokenizer.pad_token_id
    hubert_config.bos_token_id = tokenizer.bos_token_id
    hubert_config.eos_token_id = tokenizer.eos_token_id

    if config.get("use_self_attention", False):
        from model_sa import FillerHubertSAModel
        ModelClass = FillerHubertSAModel
        hubert_config.num_sa_layers = config.get("num_sa_layers", 4)
        hubert_config.sa_nhead = config.get("sa_nhead", hubert_config.num_attention_heads)
        hubert_config.sa_dim_feedforward = config.get("sa_dim_feedforward", hubert_config.intermediate_size)
        hubert_config.sa_dropout = config.get("sa_dropout", hubert_config.final_dropout)
        hubert_config.use_sinusoidal_pe = config.get("use_sinusoidal_pe", False)
        hubert_config.pe_scale = config.get("pe_scale", 1.0)
        print(f"Using FillerHubertSAModel: {hubert_config.num_sa_layers} self-attention layers "
              f"(nhead={hubert_config.sa_nhead}, ff={hubert_config.sa_dim_feedforward}); "
              f"sinusoidal_pe={hubert_config.use_sinusoidal_pe} (scale={hubert_config.pe_scale})")
    else:
        ModelClass = FillerHubertModel

    model = ModelClass.from_pretrained(
        model_checkpoint, config=hubert_config, ignore_mismatched_sizes=True,
    )
    freeze_fe = config.get("freeze_feature_extractor", True)
    if freeze_fe:
        model.freeze_feature_extractor()

    # 5. Save base config/extractor/tokenizer on main process only
    if training_args.local_rank <= 0:
        model.hubert.save_pretrained(os.path.join(exp_dir, "model"))
        feature_extractor.save_pretrained(exp_dir)
        tokenizer.save_pretrained(exp_dir)

    data_collator = DataCollatorLazyFillerASR(feature_extractor, tokenizer)
    compute_metrics = get_compute_metrics_fn(tokenizer)

    trainer = FillerASRTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        lr_schedule_type=config.get("lr_schedule_type", "triphase"),
        lr_schedule_ratios=config.get("lr_schedule_ratios", [0.1, 0.4, 0.5]),
        classifier_only_train_ratio=config.get("classifier_only_train_ratio", 0.1),
        freeze_feature_extractor=freeze_fe,
    )

    # resume if a checkpoint already exists in exp_dir
    resume = None
    ckpts = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")] if os.path.isdir(exp_dir) else []
    if ckpts:
        resume = max([os.path.join(exp_dir, d) for d in ckpts], key=os.path.getmtime)
        print(f"Resuming from {resume}")

    trainer.train(resume_from_checkpoint=resume)
    print("TRAIN_LOCAL_DONE")


if __name__ == "__main__":
    main()
