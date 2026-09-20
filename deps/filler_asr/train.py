import os
import sys
import json
import argparse
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    HubertConfig,
    HubertModel,
)
from dataset import get_dataset, extract_vocab
from collator import DataCollatorForFillerASR
from model import FillerASRTrainer, FillerHubertModel
from utils import get_compute_metrics_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    out_dir = config.get("output_dir", "experiments")
    exp_name = config.get("exp_name", "filler_asr_960h")
    wandb_project = config.get("wandb_project", "filler_asr")
    os.environ["WANDB_PROJECT"] = wandb_project
    
    dataset_name = config.get("dataset_name", "librispeech_asr")
    num_proc = config.get("num_processes", 8)
    model_checkpoint = config.get("model_checkpoint", "facebook/hubert-base-ls960")

    exp_dir = os.path.join(out_dir, exp_name)

    # 1. Initialize Training Arguments early so we can use distributed state
    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        eval_strategy="steps",
        num_train_epochs=config.get("num_train_epochs", 100),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True,
        save_steps=config.get("save_steps", 500),
        eval_steps=config.get("eval_steps", 500),
        logging_steps=config.get("logging_steps", 100),
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.005),
        warmup_steps=config.get("warmup_steps", 0),
        save_total_limit=2,
        max_steps=args.max_steps if args.max_steps != -1 else config.get("max_steps", -1),
        report_to="wandb",
        run_name=exp_name,
        dataloader_num_workers=num_proc,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        remove_unused_columns=False,
    )

    # Initialize Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    # 2. Extract vocabulary on main process only
    vocab_file = os.path.join(exp_dir, "vocab.json")
    with training_args.main_process_first(desc="vocab extraction"):
        if training_args.local_rank <= 0:
            print("Loading raw datasets to extract vocabulary...")
            cache_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(cache_dir, exist_ok=True)
            from datasets import load_dataset, concatenate_datasets
            train_clean_100 = load_dataset(dataset_name, "clean", split="train.100", cache_dir=cache_dir)
            train_clean_360 = load_dataset(dataset_name, "clean", split="train.360", cache_dir=cache_dir)
            train_other_500 = load_dataset(dataset_name, "other", split="train.500", cache_dir=cache_dir)
            raw_train_dataset = concatenate_datasets([train_clean_100, train_clean_360, train_other_500])

            print("Extracting vocabulary...")
            extract_vocab(raw_train_dataset, out_dir, exp_name)

    # Initialize Tokenizer (all processes wait until vocab is created)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ['<fill>']})

    # 3. Process dataset synchronously across processes (main process maps first, others load from cache)
    with training_args.main_process_first(desc="dataset mapping"):
        print("Preparing dataset with tokenizer and feature extractor...")
        dataset = get_dataset(feature_extractor, tokenizer, num_proc=num_proc, dataset_name=dataset_name)

    print("Loading and configuring model...")
    hubert_config = HubertConfig.from_pretrained(model_checkpoint)
    hubert_config.vocab_size = len(tokenizer)
    hubert_config.pad_token_id = tokenizer.pad_token_id
    hubert_config.bos_token_id = tokenizer.bos_token_id
    hubert_config.eos_token_id = tokenizer.eos_token_id

    model = FillerHubertModel.from_pretrained(
        model_checkpoint,
        config=hubert_config,
        ignore_mismatched_sizes=True
    )
    freeze_feature_extractor = config.get("freeze_feature_extractor", True)
    if freeze_feature_extractor:
        model.freeze_feature_extractor()

    # 4. Save configs/extractors only on main process
    if training_args.local_rank <= 0:
        model.hubert.save_pretrained(os.path.join(exp_dir, "model"))
        feature_extractor.save_pretrained(exp_dir)
        tokenizer.save_pretrained(exp_dir)

    data_collator = DataCollatorForFillerASR(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
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
        classifier_only_train_ratio=config.get("classifier_only_train_ratio", 0.0),
        freeze_feature_extractor=freeze_feature_extractor,
    )
    
    # Detect checkpoint
    checkpoint = None
    if os.path.isdir(exp_dir):
        checkpoints = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint = max([os.path.join(exp_dir, d) for d in checkpoints], key=os.path.getmtime)
            print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == "__main__":
    main()
