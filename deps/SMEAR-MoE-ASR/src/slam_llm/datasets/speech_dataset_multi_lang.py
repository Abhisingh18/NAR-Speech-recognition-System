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

###CHANGE
# as Assamese - অসমীয়া, bn Bangla - বাংলা, brx Boro - बड़ो, gu Gujarati - ગુજરાતી, hi Hindi - हिंदी, kn Kannada - ಕನ್ನಡ, ks Kashmiri - كٲشُر, gom Konkani Goan - कोंकणी, mai Maithili - मैथिली, ml Malayalam - മലയാളം, mni Manipuri - ꯃꯤꯇꯩꯂꯣꯟ, mr Marathi - मराठी, ne Nepali - नेपाली, or Oriya - ଓଡ଼ିଆ, pa Panjabi - ਪੰਜਾਬੀ, sa Sanskrit - संस्कृतम्, sd Sindhi - سنڌي, si Sinhala - සිංහල, ta Tamil - தமிழ், te Telugu - తెలుగు, ur Urdu - اُردُو
_LANG_DISPLAY = {
    "ta": "Tamil",
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
}

def _lang_display(code):
    if not code:
        return None
    key = str(code).strip().lower()
    if key == "other":
        return "the spoken language"
    if "-" in key:
        parts = [_LANG_DISPLAY.get(p, p.title()) for p in key.split("-")]
        return "code-mixed " + "-".join(parts)
    return _LANG_DISPLAY.get(key, key.title())

def _lang_instruction_suffix(code):
    """Short suffix, only for code-mix codes (contain '-'). Names the
    non-English language(s) so the model knows what NOT to translate English into."""
    if not code or "-" not in str(code):
        return ""
    key = str(code).strip().lower()
    non_english = [_LANG_DISPLAY.get(p, p.title()) for p in key.split("-") if p != "en"]
    if not non_english:
        return ""
    return f" Keep English words in English script; don't translate them into {'/'.join(non_english)}."
###
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
        ###CHANGE
        self.lang_override = dataset_config.get("lang", None)
        # Rolling/live context: at inference, feed each utterance's context from
        # the PREVIOUS utterance's model hypothesis instead of the jsonl's static
        # prev_context field. Populated externally via record_hypothesis() by the
        # inference driver, one utterance at a time, in session order. Requires
        # SEQUENTIAL access (shuffle=False, num_workers=0, batch_size=1) — the
        # dataset has no way to know ordering/session membership on its own.
        self.use_live_context = dataset_config.get("use_live_context", False)
        self.live_context = {}  # index -> hypothesis text, set by record_hypothesis()
        print(f"[INIT] use_live_context = {self.use_live_context}")
        ###
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
        assert self.input_type in ["raw", "mel", "feat"], "input_type must be one of [raw, mel, feat]"
        # tap-cache: input_type=feat loads precomputed encoder taps from
        # tap_cache_dir/<language>/<key>.npy instead of reading/encoding audio.
        self.tap_cache_dir = dataset_config.get("tap_cache_dir", None)
        if self.input_type == "feat":
            assert self.tap_cache_dir is not None, "input_type=feat requires dataset_config.tap_cache_dir"

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

    ###CHANGE
    def record_hypothesis(self, index, hyp_text):
        """Call this from the inference driver right after decoding item `index`,
        so item `index + 1` (if it's mid-session, i.e. its own jsonl prev_context
        is non-empty) picks up this hypothesis as ITS prev_context on the next
        __getitem__ call. No-op if there's no next item.
        """
        next_index = index + 1
        if next_index < len(self.data_list):
            self.live_context[next_index] = hyp_text
            print(f"[record_hypothesis] stored live context for index={next_index} "
                  f"(from index={index}): {repr(hyp_text)}")
        else:
            print(f"[record_hypothesis] index={index} is the last item — nothing to feed forward.")
    ###
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)
        ###DEBUG STATEMENTS
        # print("\n" + "=" * 80)
        # print(f"[DEBUG] [Sample {index}]")
        # print(f"[DEBUG]key           : {key}")
        # print(f"[DEBUG]audio         : {audio_path}")
        # print(f"[DEBUG]language json : {data_dict.get('language', '<missing>')}")
        # print(f"[DEBUG]prev_context  : {repr(data_dict.get('prev_context', ''))}")
        # print(f"[DEBUG]target        : {repr(target)}")
        # print("=" * 80)
        ###
        if self.input_type == "feat":
            # tap-cache: load precomputed [T, encoder_dim] last-layer encoder tap.
            # No audio read / no SSL forward. audio_length (# of LLM audio slots) is
            # derived from the cached frame count so it matches the post-downsampler
            # length exactly (T // ds_rate). audio_feat flows to the model as audio_rep.
            lang = data_dict.get("language", "_")
            feat_fp = osp.join(self.tap_cache_dir, lang, key + ".npy")
            audio_feat = torch.from_numpy(np.load(feat_fp)).float()   # [T, D]
            audio_length = max(1, audio_feat.shape[0] // self.encoder_projector_ds_rate)
        else:
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
        ###DEBUG STATEMENTS
        # print("\n[STEP 1] Prompt loaded from YAML:")
        # print(repr(prompt))
        ###
        if prompt is None:
            # prompt = random.choice(self.prompt_library)
            # prompt = "Transcribe speech to text. "
            prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "
        # MaLa-ASR historical context: ALL utterances use the same prompt; fill its
        # {prev_context} slot with this sample's prev_context (empty string when the
        # sample has no prior context). .replace (not .format) so stray braces in the
        # context text can't break substitution; a no-op when there's no placeholder.
        # if self.use_history_context:
        #     static_context = data_dict.get("prev_context", "")
        #     ###CHANGE
        #     # Empty static_context in the jsonl marks a SESSION BOUNDARY (first
        #     # utterance of a conversation) — always honor that, live or not, so a
        #     # stale hypothesis from a previous session never leaks across the gap.
        #     # Otherwise, in live mode, prefer the just-recorded hypothesis for this
        #     # index if one has arrived yet; fall back to the jsonl's own field
        #     # (e.g. ground truth) if record_hypothesis() hasn't been called for it
        #     # yet (covers non-sequential/debug access, or a plain oracle-context run).
        #     used_live = False
        #     if self.use_live_context and static_context != "":
        #         if index in self.live_context:
        #             prev_context = self.live_context[index]
        #             used_live = True
        #         else:
        #             prev_context = static_context
        #     else:
        #         prev_context = static_context
        #     ###
        if self.use_history_context:
            static_context = data_dict.get("prev_context", "")
            is_new_session = data_dict.get("new_session", False)
            ###CHANGE
            # Session boundary is now an EXPLICIT flag ("new_session"), independent
            # of prev_context's text content. This lets live mode work at true
            # inference time, where there's no ground-truth prior text to fall
            # back on — only the model's own previous hypothesis.
            used_live = False
            if is_new_session:
                prev_context = ""
            elif self.use_live_context and index in self.live_context:
                prev_context = self.live_context[index]
                used_live = True
            else:
                prev_context = static_context  # oracle/ground-truth fallback if no hypothesis yet
            ###
            ###DEBUG STATEMENTS
            print("\n[STEP 3] Before {prev_context} replacement:")
            print(repr(prompt))
            print("index               =", index)
            print("static prev_context =", repr(static_context))
            print("use_live_context    =", self.use_live_context)
            print("live_context[index] available =", index in self.live_context)
            print("prev_context SOURCE =", "LIVE (model hypothesis)" if used_live else
                ("SESSION BOUNDARY (new_session)" if is_new_session else "JSONL (static/ground-truth)"))
            print("prev_context USED   =", repr(prev_context))
            ###
            prompt = prompt.replace("{prev_context}", prev_context)
            ###DEBUG STATEMENTS
            print("\nAfter {prev_context} replacement:")
            print(repr(prompt))
            ###

        ###CHANGE

        # if "{lang}" in prompt:
        if "{lang}" in prompt or "{lang_note}" in prompt:

            ###DEBUG
            # print("\n[STEP 4] Language replacement")

            # print("lang_override =", repr(self.lang_override))
            # print("language(json) =", repr(data_dict.get("language")))
            ###
            lang_code = self.lang_override or data_dict.get("language", "ta")
            lang_name = _lang_display(lang_code) or "the target language"
            ###DEBUG
            # print("lang_code =", repr(lang_code))
            # print("lang_name =", repr(lang_name))
            # print("suffix =", repr(_lang_instruction_suffix(lang_code)))
            ###
            # prompt = prompt.replace("{lang}", lang_name) + _lang_instruction_suffix(lang_code)
            prompt = prompt.replace("{lang}", lang_name)
            prompt = prompt.replace("{lang_note}", _lang_instruction_suffix(lang_code))
            ###DEBUG
            # print("\nAfter {lang} replacement:")
            # print(repr(prompt))
            ###
        ###DEBUG STATEMENTS
        # print("\n[STEP 5] Before prompt template:")
        # print(repr(prompt))
        ###
        prompt = self.prompt_template.format(prompt)
        ###DEBUG STATEMENTS
        # print("\n[STEP 6] After prompt template:")
        # print(prompt)
        ###
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        ###DEBUG STATEMENTS
        # print("\n[STEP 7] Tokenization")
        # print("prompt_length =", prompt_length)
        # print("First 40 token ids:", prompt_ids[:40])

        # try:
        #     decoded = self.tokenizer.decode(prompt_ids)
        #     print("\nDecoded prompt:")
        #     print(decoded)
        # except Exception as e:
        #     print("Could not decode:", e)
        ###
        ###CHANGE
        # Inference-time override: forces {lang} for every sample, bypassing the
        # jsonl 'language' key. Unset during training so per-sample language is used.
        # self.lang_override = dataset_config.get("lang", None)
        ###
        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_feat": audio_feat if self.input_type == "feat" else None,
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
            "audio_feat": audio_feat if self.input_type == "feat" else None,
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
        elif self.input_type == "feat":
            # tap-cache: pad precomputed taps to [B, Tmax, D] and build the
            # feature-resolution mask. Fed to the model as audio_rep (encoder skipped).
            audio_feat_max_length = max([s['audio_feat'].shape[0] for s in samples])
            audio_rep = torch.stack([self.pad(s['audio_feat'], audio_feat_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_feat_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio_feat'].shape[0]] = 1

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
                "audio_mask": audio_mask if self.input_type in ("raw", "feat") else None,
                "audio_rep": audio_rep if self.input_type == "feat" else None,
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
            "audio_mask": audio_mask if self.input_type in ("raw", "feat") else None,
            "audio_rep": audio_rep if self.input_type == "feat" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask
        }



def get_speech_dataset(dataset_config, tokenizer, split, num_lang=1, **kwargs):
    # NOTE: SMEAR-MoE's get_custom_dataset (utils/dataset_utils.py) calls this with
    # `num_lang` as a 4th positional arg. SMEAR routing is learned by the router, so the
    # dataset itself does not need to build per-language masks; num_lang is accepted and
    # ignored here for signature compatibility.
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset