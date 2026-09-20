import os
import re
import json
from datasets import load_dataset, Audio, concatenate_datasets, DatasetDict

CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"]'

def compute_output_length(input_length):
    """
    Computes the output sequence length of standard Hubert/Wav2Vec2 base models
    given the input audio length.
    """
    length = input_length
    for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        length = (length - kernel) // stride + 1
    return length

def extract_vocab(dataset, out_dir, exp_name):
    """
    Extracts vocabulary from the training dataset and saves it to a vocab.json file.
    """
    vocab_set = set()
    
    for text in dataset["text"]:
        text = re.sub(CHARS_TO_IGNORE_REGEX, '', text).lower()
        # Add characters to vocabulary
        for char in text:
            if char != ' ':
                vocab_set.add(char)
                
    vocab_dict = {char: idx for idx, char in enumerate(sorted(list(vocab_set)))}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["<unk>"] = len(vocab_dict)
    vocab_dict["<pad>"] = len(vocab_dict)
    vocab_dict["<s>"] = len(vocab_dict)
    vocab_dict["</s>"] = len(vocab_dict)
    
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    vocab_file = os.path.join(exp_dir, "vocab.json")
    
    with open(vocab_file, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
        
    return vocab_file

def get_dataset(feature_extractor, tokenizer, num_proc=8, dataset_name="librispeech_asr"):
    cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load the 960h LibriSpeech dataset
    # It contains train.clean.100, train.clean.360, train.other.500
    train_clean_100 = load_dataset(dataset_name, "clean", split="train.100", cache_dir=cache_dir)
    train_clean_360 = load_dataset(dataset_name, "clean", split="train.360", cache_dir=cache_dir)
    train_other_500 = load_dataset(dataset_name, "other", split="train.500", cache_dir=cache_dir)
    
    train_dataset = concatenate_datasets([train_clean_100, train_clean_360, train_other_500])
    test_dataset = load_dataset(dataset_name, "clean", split="validation", cache_dir=cache_dir) # dev-clean
    
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    def remove_special_characters(batch):
        batch["text"] = re.sub(CHARS_TO_IGNORE_REGEX, '', batch["text"]).lower()
        return batch

    dataset = dataset.map(remove_special_characters, num_proc=num_proc)
    
    # Resample audio if necessary
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        # Output length of the model (number of frames T)
        T = compute_output_length(batch["input_length"])
        batch["output_length"] = T
        
        # Prepare text tokens
        text = batch["text"].replace(" ", "|")
        text_tokens = tokenizer(text).input_ids
        
        # We need tokens: [<s>, char_1, ..., char_N, </s>, <fill>, <fill>, ...] up to T
        fill_id = tokenizer.convert_tokens_to_ids("<fill>")
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        
        if len(text_tokens) + 2 > T:
            text_tokens = text_tokens[:T-2]
            
        labels = [bos_id] + text_tokens + [eos_id]
        
        fill_length = T - len(labels)
        labels = labels + [fill_id] * fill_length
        
        batch["labels"] = labels
        
        return batch

    # Need to remove all existing columns to avoid conflicts with tokenized format
    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=num_proc)
    
    return dataset
