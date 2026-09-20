import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, HubertPreTrainedModel, HubertModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.trainer import is_optimizer_factory, is_sagemaker_mp_enabled

class FillerHubertModel(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hubert = HubertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

    def freeze_all_except_classifier(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def unfreeze_all_except_feature_extractor(self, freeze_feature_extractor=True):
        for param in self.parameters():
            param.requires_grad = True
        if freeze_feature_extractor:
            self.freeze_feature_extractor()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutput(
            loss=None, # loss is computed in Trainer
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FillerASRTrainer(Trainer):
    def __init__(
        self,
        *args,
        lr_schedule_type="linear",
        lr_schedule_ratios=None,
        classifier_only_train_ratio=0.0,
        freeze_feature_extractor=True,
        lr_exp_final_ratio=0.01,
        fill_weight=1.0,
        fill_token_id=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_ratios = self._validate_triphase_ratios(lr_schedule_ratios or [0.1, 0.4, 0.5])
        self.classifier_only_train_ratio = float(classifier_only_train_ratio)
        self.freeze_feature_extractor = freeze_feature_extractor
        self.lr_exp_final_ratio = float(lr_exp_final_ratio)
        # framewise-CE class weight: down-weight the <fill> majority class so that
        # grading the full frame budget (frames_per_sec=50) doesn't drown the loss.
        self.fill_weight = float(fill_weight)
        self.fill_token_id = fill_token_id
        self._ce_weight = None
        self._freeze_stage = None

    @staticmethod
    def _validate_triphase_ratios(ratios):
        if len(ratios) != 3:
            raise ValueError("lr_schedule_ratios must contain [warmup, constant, decay] ratios.")
        ratios = tuple(float(ratio) for ratio in ratios)
        if any(ratio < 0 for ratio in ratios):
            raise ValueError("lr_schedule_ratios cannot contain negative values.")
        total = sum(ratios)
        if total <= 0:
            raise ValueError("lr_schedule_ratios must sum to a positive value.")
        return tuple(ratio / total for ratio in ratios)

    @staticmethod
    def _get_triphase_phase_steps(num_training_steps, ratios):
        if num_training_steps <= 0:
            return 0, 0, 0

        warmup_ratio, constant_ratio, _ = ratios
        warmup_steps = int(round(num_training_steps * warmup_ratio))
        constant_steps = int(round(num_training_steps * constant_ratio))

        if warmup_ratio > 0 and warmup_steps == 0:
            warmup_steps = 1
        if constant_ratio > 0 and constant_steps == 0 and warmup_steps < num_training_steps:
            constant_steps = 1

        warmup_steps = min(warmup_steps, num_training_steps)
        constant_steps = min(constant_steps, num_training_steps - warmup_steps)
        decay_steps = max(0, num_training_steps - warmup_steps - constant_steps)
        return warmup_steps, constant_steps, decay_steps

    @classmethod
    def get_triphase_lr_lambda(cls, num_training_steps, ratios):
        warmup_steps, constant_steps, decay_steps = cls._get_triphase_phase_steps(num_training_steps, ratios)

        def lr_lambda(current_step):
            if num_training_steps <= 0:
                return 1.0
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < warmup_steps + constant_steps:
                return 1.0
            if decay_steps == 0:
                return 0.0
            decay_step = current_step - warmup_steps - constant_steps
            return max(0.0, float(decay_steps - decay_step) / float(max(1, decay_steps)))

        return lr_lambda

    @staticmethod
    def get_exponential_lr_lambda(num_training_steps, warmup_steps, final_ratio):
        """Linear warmup, then geometric (exponential) decay from peak LR down to
        `final_ratio` * peak at the last step."""
        warmup_steps = max(0, min(int(warmup_steps), num_training_steps))
        decay_steps = max(1, num_training_steps - warmup_steps)
        final_ratio = max(1e-8, float(final_ratio))
        gamma = final_ratio ** (1.0 / decay_steps)

        def lr_lambda(current_step):
            if num_training_steps <= 0:
                return 1.0
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(final_ratio, gamma ** (current_step - warmup_steps))

        return lr_lambda

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def _log_stage_change(self, message):
        if self.args.local_rank in (-1, 0):
            print(message)

    def _set_freezing_stage(self, model, stage):
        base_model = self._unwrap_model(model)
        if self._freeze_stage == stage:
            return

        if stage == "classifier_only":
            base_model.freeze_all_except_classifier()
            self._log_stage_change("Freezing model: training lm_head only.")
        elif stage == "full_except_feature_extractor":
            base_model.unfreeze_all_except_feature_extractor(self.freeze_feature_extractor)
            if self.freeze_feature_extractor:
                self._log_stage_change("Unfreezing model: training all layers except hubert.feature_extractor.")
            else:
                self._log_stage_change("Unfreezing model: training all layers.")
        else:
            raise ValueError(f"Unknown freezing stage: {stage}")

        self._freeze_stage = stage
        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()

    def _maybe_update_freezing_stage(self, model):
        if self.classifier_only_train_ratio <= 0:
            self._set_freezing_stage(model, "full_except_feature_extractor")
            return

        if self.state.max_steps <= 0:
            self._set_freezing_stage(model, "classifier_only")
            return

        classifier_only_steps = int(math.ceil(self.state.max_steps * self.classifier_only_train_ratio))
        if self.state.global_step < classifier_only_steps:
            self._set_freezing_stage(model, "classifier_only")
        else:
            self._set_freezing_stage(model, "full_except_feature_extractor")

    def create_optimizer(self, model=None):
        opt_model = self.model if model is None else model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if is_optimizer_factory(optimizer_cls):
                self.optimizer = optimizer_cls()(opt_model, **optimizer_kwargs)
            else:
                if "params" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("params")
                if "model" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("model")
                if "optimizer_dict" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            from transformers.trainer import smp

            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None and self.lr_schedule_type == "triphase":
            optimizer = optimizer if optimizer is not None else self.optimizer
            self.lr_scheduler = LambdaLR(
                optimizer,
                self.get_triphase_lr_lambda(num_training_steps, self.lr_schedule_ratios),
            )
            self._created_lr_scheduler = True
            warmup_steps, constant_steps, decay_steps = self._get_triphase_phase_steps(
                num_training_steps,
                self.lr_schedule_ratios,
            )
            self._log_stage_change(
                "Using triphase LR scheduler: "
                f"warmup={warmup_steps}, constant={constant_steps}, decay={decay_steps} steps."
            )
            return self.lr_scheduler

        if self.lr_scheduler is None and self.lr_schedule_type == "exponential":
            optimizer = optimizer if optimizer is not None else self.optimizer
            self.lr_scheduler = LambdaLR(
                optimizer,
                self.get_exponential_lr_lambda(
                    num_training_steps, self.args.warmup_steps, self.lr_exp_final_ratio
                ),
            )
            self._created_lr_scheduler = True
            warmup = max(0, min(int(self.args.warmup_steps), num_training_steps))
            self._log_stage_change(
                "Using exponential LR scheduler: "
                f"warmup={warmup}, then geometric decay to {self.lr_exp_final_ratio:g} of peak "
                f"over {num_training_steps} steps."
            )
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer=optimizer)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._maybe_update_freezing_stage(model)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Overrides the standard compute_loss to use CrossEntropyLoss instead of CTCLoss.
        """
        labels = inputs.pop("labels")
        
        # We don't want the model to automatically compute loss (which defaults to CTC)
        # So we pass inputs without labels.
        outputs = model(**inputs)

        logits = outputs.logits

        # optional per-class weight: down-weight the <fill> majority class
        weight = None
        if self.fill_weight != 1.0 and self.fill_token_id is not None:
            if self._ce_weight is None or self._ce_weight.numel() != logits.size(-1):
                w = torch.ones(logits.size(-1))
                w[self.fill_token_id] = self.fill_weight
                self._ce_weight = w
            weight = self._ce_weight.to(device=logits.device, dtype=logits.dtype)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)

        # logits: (batch_size, sequence_length, vocab_size)
        # labels: (batch_size, sequence_length)
        
        # In case the sequence lengths don't exactly match due to downsampling approximations,
        # we truncate to the minimum length to be safe. But they should match exactly based on our formula.
        min_seq_len = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :min_seq_len, :].contiguous()
        labels = labels[:, :min_seq_len].contiguous()
        
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
