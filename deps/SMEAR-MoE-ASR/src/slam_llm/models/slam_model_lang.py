def forward(self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
            ):
    audio_mel = kwargs.get("audio_mel", None)
    audio_mel_mask = kwargs.get("audio_mel_mask", None)
    audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper

    audio = kwargs.get("audio", None)
    audio_mask = kwargs.get("audio_mask", None)
    visual = kwargs.get("visual", None)
    visual_mask = kwargs.get("visual_mask", None)
    text = kwargs.get("text", None)

    # for text encoder
    instruct_ids = kwargs.get("instruct_ids", None)
    instruct_mask = kwargs.get("instruct_mask", None)

    modality_mask = kwargs.get("modality_mask", None)
    
    zh_data = kwargs.get("zh", None)
    en_data = kwargs.get("en", None)

    # Get language masks for each of the 4 languages
    lang1_mask = kwargs.get("lang1_mask", None)  # B x 1
    lang2_mask = kwargs.get("lang2_mask", None)  # B x 1
    lang3_mask = kwargs.get("lang3_mask", None)  # B x 1
    lang4_mask = kwargs.get("lang4_mask", None)  # B x 1

    encoder_outs = None
    if audio_mel is not None or audio is not None or visual is not None or text is not None:
        if self.train_config.freeze_encoder: # freeze encoder
            self.encoder.eval()

        if self.model_config.encoder_name == "whisper":
            if self.model_config.encoder_path_hf is not None:
                # encoder_outs = self.encoder(audio_mel.to(dtype=torch.bfloat16).permute(0, 2, 1))['last_hidden_state'] # bs*seq*dim
                encoder_outs = self.encoder(audio_mel.permute(0, 2, 1))['last_hidden_state'] # bs*seq*dim
                
            elif self.model_config.encoder_path.endswith('.pt'):
                encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
            else:
                if not hasattr(self, 'encoder') or self.encoder is None:
                    from transformers import WhisperForConditionalGeneration

                    # Load the full model from Hugging Face directory
                    model = WhisperForConditionalGeneration.from_pretrained(self.model_config.encoder_path)

                    # Extract the encoder and assign it to self.encoder
                    self.encoder = model.model.encoder

                # Use self.encoder directly
                encoder_outs = self.encoder(audio_mel.to(dtype=torch.float32).permute(0, 2, 1))['last_hidden_state']
        if self.model_config.encoder_name == "beats":
            encoder_outs, audio_mel_post_mask = self.encoder.extract_features(audio_mel, audio_mel_mask) # bs*seq*dim
        if self.model_config.encoder_name == "eat":
            encoder_outs = self.encoder.model.extract_features(audio_mel.unsqueeze(dim=1), padding_mask = None, mask=False, remove_extra_tokens = False)['x']
        if self.model_config.encoder_name == "clap": 
            if text is not None: 
                encoder_outs = self.encoder.encode_text(text).unsqueeze(1)  # [btz, 1, dim]        
            elif audio is not None: 
                encoder_outs = self.encoder.encode_audio(audio)  # with projection-based decoding 
        if self.model_config.encoder_name == "SpatialAST":
            encoder_outs = self.encoder(audio) # output: [bs, seq_len=3+512, dim=768]
        if self.model_config.encoder_name == "wavlm":
            encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask) #(FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
        if self.model_config.encoder_name == "hubert":
            results = self.encoder(source = audio, padding_mask = 1-audio_mask) 
            if self.model_config.encoder_type == "pretrain":
                encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
            if self.model_config.encoder_type == "finetune":
                encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                print("hubert encoder out", encoder_outs)
                encoder_outs = encoder_outs.transpose(0, 1)
        if self.model_config.encoder_name == "av_hubert":
            results = self.encoder(source={'video':visual, 'audio':audio}, padding_mask=visual_mask) # bs*seq*dim  
            encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
            encoder_outs = encoder_outs.transpose(0, 1)
            audio_mel_post_mask = (~audio_mel_post_mask).float()
        if self.model_config.encoder_name == 'musicfm':
            encoder_outs = self.encoder.extract_features(audio, padding_mask = None) # MusicFM doesn't support padding mask 
        if self.model_config.encoder_name == "emotion2vec":
            encoder_outs = self.encoder.extract_features(audio, None)['x'] # bs*seq*dim
        if self.encoder is None:
            encoder_outs = audio_mel if audio_mel is not None else audio

        if self.model_config.encoder_projector == "q-former":
            encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
        if self.model_config.encoder_projector == "linear":
            # encoder_outs = encoder_outs.to(torch.float32)  # Convert to match projector dtype for gemma and whisper whisper output float16, gemma need float32
            encoder_outs = self.encoder_projector(encoder_outs)
        
        # Using four separate conv1d-linear projectors for different languages
        if self.model_config.encoder_projector == "cov1d-linear":
            # Create a new tensor to hold the projected outputs
            batch_size = encoder_outs.shape[0]
            seq_len = encoder_outs.shape[1]
            hidden_dim = self.encoder_projector_lang1.output_dim  # Assuming all projectors have the same output dimension
            
            # Initialize with zeros
            projected_outs = torch.zeros((batch_size, seq_len, hidden_dim), device=encoder_outs.device)
            
            # Apply each language-specific projector to the entire batch, then use the masks to select
            # the appropriate outputs for each sample in the batch
            if lang1_mask is not None and lang1_mask.sum() > 0:
                lang1_projected = self.encoder_projector_lang1(encoder_outs)
                lang1_indices = torch.where(lang1_mask.squeeze(-1))[0]
                projected_outs[lang1_indices] = lang1_projected[lang1_indices]
                
            if lang2_mask is not None and lang2_mask.sum() > 0:
                lang2_projected = self.encoder_projector_lang2(encoder_outs)
                lang2_indices = torch.where(lang2_mask.squeeze(-1))[0]
                projected_outs[lang2_indices] = lang2_projected[lang2_indices]
                
            if lang3_mask is not None and lang3_mask.sum() > 0:
                lang3_projected = self.encoder_projector_lang3(encoder_outs)
                lang3_indices = torch.where(lang3_mask.squeeze(-1))[0]
                projected_outs[lang3_indices] = lang3_projected[lang3_indices]
                
            if lang4_mask is not None and lang4_mask.sum() > 0:
                lang4_projected = self.encoder_projector_lang4(encoder_outs)
                lang4_indices = torch.where(lang4_mask.squeeze(-1))[0]
                projected_outs[lang4_indices] = lang4_projected[lang4_indices]
            
            # Update encoder_outs with our language-specific projections
            encoder_outs = projected_outs

    if instruct_ids is not None:
        if self.encoder is not None:
            encoder_outs = self.encoder(input_ids=instruct_ids, attention_mask=instruct_mask).last_hidden_state
        if self.model_config.encoder_projector == "q-former":
            encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
        if self.model_config.encoder_projector == "linear":
            encoder_outs = self.encoder_projector(encoder_outs)
            

    if input_ids is not None:
        input_ids[input_ids == -1] = 0
        if isinstance(self.llm, T5ForConditionalGeneration):
            inputs_embeds = self.llm.shared(input_ids)
        else:
            if hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

    if modality_mask is not None:
        modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
        modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

        encoder_outs_pad = torch.zeros_like(inputs_embeds)
        for i in range(encoder_outs.shape[0]):
            encoder_outs_pad[
                i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
            ] = encoder_outs[i][:modality_lengths[i]]
        
        inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

    if kwargs.get("inference_mode", False):
        return inputs_embeds, attention_mask

    if zh_data is not None and en_data is not None:
        model_outputs, acc = self.llm(zh=zh_data, en=en_data)
    else:
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

    return model_outputs, acc






# Inside your model's __init__ method

# Encoder projectors for each language
if self.model_config.encoder_projector == "cov1d-linear":
    self.encoder_projector_lang1 = Conv1DProjector(
        input_dim=self.encoder_dim,
        output_dim=self.llm_dim,
        kernel_size=3,
        padding=1
    )
    self.encoder_projector_lang2 = Conv1DProjector(
        input_dim=self.encoder_dim,
        output_dim=self.llm_dim,
        kernel_size=3,
        padding=1
    )
    self.encoder_projector_lang3 = Conv1DProjector(
        input_dim=self.encoder_dim,
        output_dim=self.llm_dim,
        kernel_size=3,
        padding=1
    )
    self.encoder_projector_lang4 = Conv1DProjector(
        input_dim=self.encoder_dim,
        output_dim=self.llm_dim,
        kernel_size=3,
        padding=1
    )