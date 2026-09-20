import torch
import soundfile as sf
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

class WhisperDecoderExtractor:
    def __init__(self, model_path, lang_code="hi", device="cuda:7"):
        # Load pipeline for convenience (tokenizer + config setup)
        self.whisper_asr = pipeline(
            "automatic-speech-recognition", model=model_path, device=7
        )
        print("model half loaded")
        # Ensure language forcing works like your snippet
        if lang_code == "or":
            self.whisper_asr.model.config.forced_decoder_ids = (
                self.whisper_asr.tokenizer.get_decoder_prompt_ids(
                    language=None, task="transcribe"
                )
            )
        else:
            self.whisper_asr.model.config.forced_decoder_ids = (
                self.whisper_asr.tokenizer.get_decoder_prompt_ids(
                    language=lang_code, task="transcribe"
                )
            )

        print("model loaded!!!!")
        # Keep raw model + processor for hidden states
        self.model: WhisperForConditionalGeneration = self.whisper_asr.model
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.device = device
        self.model.to(device).eval()

        print(f"IndicWhisper loaded from {model_path} with lang={lang_code}")

    def forward(self, audio_path, max_length=225):
        """
        Input:
            audio_path: path to .wav/.mp3 file
        Output:
            decoder_states: Tensor (seq_len, hidden_dim)
            seq_len: int
        Prints generated transcript.
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != self.processor.feature_extractor.sampling_rate:
            raise ValueError(f"Expected {self.processor.feature_extractor.sampling_rate} Hz, got {sr}")

        # Convert to features
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)

        # Encode
        encoder_out = self.model.model.encoder(inputs)

        # Generate transcript autoregressively
        with torch.no_grad():
            gen_tokens = self.model.generate(
                inputs=encoder_out[0],
                max_length=max_length,
                do_sample=False
            )

            # Run decoder again with generated tokens to extract hidden states
            decoder_out = self.model.model.decoder(
                input_ids=gen_tokens,
                encoder_hidden_states=encoder_out[0]
            )

        decoder_states = decoder_out.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        seq_len = decoder_states.size(0)

        # Decode transcript
        transcript = self.processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"Transcript: {transcript}")

        return decoder_states, seq_len


model_path = "/raid/isha/SpeechLLM/SLAM-LLM/models_pretrained/audio_encoder/whisper/whisper-medium-hi_alldata_multigpu"
whisper_decoder = WhisperDecoderExtractor(model_path)
decoder_states, seq_len = whisper_decoder("/nfs/projects/DATA-MARCH/fleurs/hindi/wavs-16k/train/16970453014023876157.wav")