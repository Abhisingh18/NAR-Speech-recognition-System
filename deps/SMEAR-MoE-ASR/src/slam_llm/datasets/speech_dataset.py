import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
try:
    import whisper as _whisper_mod
except ImportError:
    _whisper_mod = None  # openai-whisper not installed; raw input_type still works
from slam_llm.utils.compute_utils import calculate_output_length_1d


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        # MaLa-ASR historical context (see examples/asr_librispeech): when
        # use_history_context is set, EVERY utterance uses the same prompt; its
        # {prev_context} slot is filled per-sample from prev_context (empty string
        # when the sample has no prior context). No per-sample prompt selection.
        self.use_history_context = dataset_config.get("use_history_context", False)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        # self.prompt_library = [
        #     "Begin by converting the spoken words into written text. ",
        #     "Can you transcribe the speech into a written format? ",
        #     "Focus on translating the audible content into text. ",
        #     "Transcribe the speech by carefully listening to it. ",
        #     "Would you kindly write down the content of the speech? ",
        #     "Analyze the speech and create a written transcription. ",
        #     "Engage with the speech to produce a text-based version. ",
        #     "Can you document the speech in written form? ",
        #     "Transform the spoken words into text accurately. ",
        #     "How about putting the speech's content into writing? "
        # ]
        _prompt_style = dataset_config.get("prompt_style", "vicuna")
        if _prompt_style == "gemma2":
            self.prompt_template = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            if self.tokenizer is not None and "<end_of_turn>" in self.tokenizer.all_special_tokens:
                self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            else:
                self.eot_token_id = None
        elif _prompt_style == "qwen2":
            self.prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.eot_token_id = None
        elif _prompt_style == "airavata":
            # ai4bharat/Airavata: tulu-style chat template. Tokenizer prepends <s> BOS
            # automatically; answer is terminated with </s> via tokenizer.eos_token_id,
            # so no custom eot is needed here.
            self.prompt_template = "<|user|>\n{}\n<|assistant|>\n"
            self.eot_token_id = None
        else:
            self.prompt_template = "USER: {}\n ASSISTANT:"
            self.eot_token_id = None
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        # Frame-budget mode: when audio_frames_per_sec > 0 (raw input only), reserve
        # audio token slots from DURATION as round(fps * sec) // ds_rate instead of
        # the full encoder frame rate (samples//320 = 50 frames/s). The scatter in
        # slam_model then fills only these slots with the LEFTMOST encoder frames,
        # so the <fill>-heavy tail beyond the budget never reaches the LLM. ds_rate
        # here must match model_config.encoder_projector_ds_rate.
        self.audio_frames_per_sec = dataset_config.get("audio_frames_per_sec", -1.0)
        self.encoder_projector_ds_rate = dataset_config.get("encoder_projector_ds_rate", 5)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

        # Precompute a per-utterance length for optional length-bucketed batching
        # (enabled via train_config.sort_by_length). Portable across datasets, in
        # priority order:
        #   1) an explicit "source_len" field in the manifest (fastest, recommended;
        #      bake it into any jsonl with scripts/add_source_len.py);
        #   2) a duration embedded in the key as _<start9>_<end9>_ (ms), as used by
        #      the Indic-ASR manifests -- pure string op, no audio IO;
        #   3) the audio header via soundfile.info (header only, no decode) -- correct
        #      for ANY dataset but slow on large sets, so we log a one-time hint.
        # Only the RELATIVE order matters for sorting; a given manifest resolves via
        # one branch consistently, so units never mix in practice.
        import re as _re
        _KEY_LEN = _re.compile(r'_(\d{9})_(\d{9})_')
        self.speech_lengths = []
        _hdr_reads = 0
        for _d in self.data_list:
            _n = _d.get("source_len")
            if _n is None:
                _m = _KEY_LEN.search(_d.get("key", ""))
                if _m:
                    _n = int(_m.group(2)) - int(_m.group(1))
                else:
                    try:
                        _n = int(sf.info(_d["source"]).frames)
                        _hdr_reads += 1
                    except Exception:
                        _n = 0
            self.speech_lengths.append(_n)
        if _hdr_reads:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "[sort_by_length] read audio headers for %d/%d utterances "
                "(no source_len / key duration). Precompute 'source_len' via "
                "scripts/add_source_len.py to avoid this at startup.",
                _hdr_reads, len(self.data_list),
            )

        # # debug
        # with open(dataset_config.train_data_path, encoding='utf-8') as fin:
        #         for line in fin:
        #             data_dict = json.loads(line.strip())
        #             self.data_list.append(data_dict)
        # if split == "train":
        #     self.data_list = self.data_list[:80]
        # else:
        #     self.data_list = self.data_list[80:100]

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
    
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)

        if _whisper_mod is not None:
            audio_raw = _whisper_mod.load_audio(audio_path)
        else:
            # fallback when openai-whisper is not installed (e.g. triton conflict)
            import soundfile as _sf, numpy as _np
            _audio, _sr = _sf.read(audio_path, dtype='float32', always_2d=False)
            if _audio.ndim > 1:
                _audio = _audio.mean(axis=1)
            if _sr != 16000:
                import torchaudio.functional as _F_ta
                _audio = _F_ta.resample(torch.from_numpy(_audio), _sr, 16000).numpy()
            audio_raw = _audio
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            if self.audio_frames_per_sec > 0:
                # duration-based frame budget: keep only the leftmost
                # round(fps * sec) encoder frames (e.g. fps=15 of the 50/s)
                seconds = len(audio_raw) / 16000.0
                budget_frames = int(round(self.audio_frames_per_sec * seconds))
                audio_length = max(1, budget_frames // self.encoder_projector_ds_rate)
            else:
                audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
                audio_length = audio_length // self.encoder_projector_ds_rate # fc downsample (historically 5)
        elif self.input_type == "mel":
            if _whisper_mod is None:
                raise ImportError(
                    'openai-whisper is required for input_type=mel. '
                    'Install it manually (note: incompatible with torch>=2.4+cu12x).')
            audio_raw = _whisper_mod.pad_or_trim(audio_raw)
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = _whisper_mod.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        prompt = self.prompt
        if prompt is None:
            # prompt = random.choice(self.prompt_library)
            # prompt = "Transcribe speech to text. "
            prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "
        # MaLa-ASR historical context: ALL utterances use the same prompt; fill its
        # {prev_context} slot with this sample's prev_context (empty string when the
        # sample has no prior context). .replace (not .format) so stray braces in the
        # context text can't break substitution; a no-op when there's no placeholder.
        if self.use_history_context:
            prev_context = data_dict.get("prev_context", "")
            prompt = prompt.replace("{prev_context}", prev_context)
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        answer = self.answer_template.format(target)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64
        )
        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples] #[120, 48, 82, 42]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]  #[0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])


        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask
        }



def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
