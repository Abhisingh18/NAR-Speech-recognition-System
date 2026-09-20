import math
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

# duration-based frame budget: only the first ceil(dur * frames_per_sec) frames
# contribute to the framewise CE loss; the rest (the <fill> tail) are ignored.
# frames_per_sec=25 grades ~half the HuBERT frames (encoder runs at ~50 fps);
# frames_per_sec=50 grades ~all frames (n_keep >= T, only padding stays -100).
SAMPLING_RATE = 16000
FRAMES_PER_SEC = 25

@dataclass
class DataCollatorForFillerASR:
    feature_extractor: any
    tokenizer: any
    padding: Union[bool, str] = True
    frames_per_sec: int = FRAMES_PER_SEC

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # duration frame-budget mask: keep only the first n_keep frames per
        # utterance, ignore the rest of the <fill> tail in the loss.
        for i, feature in enumerate(features):
            n_keep = math.ceil(len(feature["input_values"]) / SAMPLING_RATE * self.frames_per_sec)
            labels[i, n_keep:] = -100

        batch["labels"] = labels
        return batch
