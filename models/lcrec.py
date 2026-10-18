"""Pretrained causal LM + LoRA shared by LCRec and its RQ-KMeans SID variant."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class LCRecOutput(NamedTuple):
    loss: torch.Tensor
    ce_per_example: torch.Tensor
    align_loss: torch.Tensor


class LCRecGeneration(NamedTuple):
    sequences: torch.Tensor


class LCRec(nn.Module):
    def __init__(self, backbone, sid_token_ids, eos_token_id, pad_token_id,
                 catalogue_sids, align_dim=None, align_source="last_inp",
                 align_loss_type="cos", cos_loss_weight=0.5):
        super().__init__()
        if align_source not in ("last_inp", "last_out", "mean_out"):
            raise ValueError("align_source must be last_inp, last_out or mean_out")
        if align_loss_type not in ("cos", "gram", "cosgram"):
            raise ValueError("align_loss_type must be cos, gram or cosgram")
        if not 0 <= cos_loss_weight <= 1:
            raise ValueError("cos_loss_weight must be finite and in [0, 1]")
        self.cos_loss_weight = cos_loss_weight
        self.backbone = backbone
        self.register_buffer("sid_token_ids", sid_token_ids.clone())
        self.sem_id_dim = sid_token_ids.shape[0]
        self.eos_token_id, self.pad_token_id = eos_token_id, pad_token_id
        self.align_source, self.align_loss_type = align_source, align_loss_type
        if align_dim is not None:
            hidden_dim = backbone.config.hidden_size
            if align_loss_type in ("cos", "cosgram"):
                self.hidden_to_item_emb = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                    nn.Linear(hidden_dim, align_dim, bias=False),
                )
            if align_loss_type in ("gram", "cosgram"):
                self.hidden_to_gram = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.has_alignment = align_dim is not None
        self.prefix_tokens = {}
        for sid in catalogue_sids:
            if len(sid) != self.sem_id_dim:
                raise ValueError("Catalogue SID length does not match sem_id_dim")
            if any(not 0 <= int(code) < sid_token_ids.shape[1] for code in sid):
                raise ValueError("Catalogue SID code outside the codebook")
            token_ids = sid_token_ids[torch.arange(self.sem_id_dim), sid].tolist()
            for index, token in enumerate(token_ids):
                self.prefix_tokens.setdefault(tuple(token_ids[:index]), set()).add(token)
        if not self.prefix_tokens:
            raise ValueError("The SID catalogue must not be empty")
        self.prefix_tokens = {key: sorted(value) for key, value in self.prefix_tokens.items()}

    def forward(self, batch):
        prompt_ids, targets = batch["input_ids"], batch["labels"]
        if targets.ndim != 2 or targets.shape[1] != self.sem_id_dim:
            raise ValueError("Targets must be complete SID token sequences")
        # The separator predicts SID[0]; supervise only SID prediction positions.
        # The final SID's next-token logits (EOS), prompt and padding are excluded.
        input_ids = torch.cat((prompt_ids, targets), dim=1)
        mask = torch.cat((batch["attention_mask"], torch.ones_like(targets)), dim=1)
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        use_alignment = self.training and self.has_alignment
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=mask, position_ids=positions,
            use_cache=False, output_hidden_states=use_alignment, return_dict=True,
            logits_to_keep=self.sem_id_dim + 1,
        )
        logits = outputs.logits[:, -(self.sem_id_dim + 1):-1].float()
        per_example = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view_as(targets).mean(dim=1)
        align_loss = per_example.new_zeros(())
        if use_alignment:
            target = batch.get("align_target_emb")
            if target is None:
                raise ValueError("Alignment requires align_target_emb")
            hidden = outputs.hidden_states[-1]
            separator = prompt_ids.shape[1] - 1
            if self.align_source == "last_inp":
                selected = hidden[:, separator - 1]
            elif self.align_source == "last_out":
                selected = hidden[:, separator]
            else:
                selected = hidden[:, separator:separator + self.sem_id_dim].mean(dim=1)
            target = target.detach().float()
            if target.ndim != 2 or target.shape[0] != len(targets) or not torch.isfinite(target).all():
                raise ValueError("Alignment requires finite vectors, one per sample")
            if self.align_loss_type in ("gram", "cosgram"):
                prediction = self.hidden_to_gram(selected.float()).float()
                with torch.autocast(device_type=selected.device.type, enabled=False):
                    normalized_prediction = F.normalize(prediction, dim=-1)
                    reference = F.normalize(target, dim=-1)
                    gram_loss = F.mse_loss(
                        normalized_prediction @ normalized_prediction.T, reference @ reference.T
                    )
                align_loss = gram_loss
            if self.align_loss_type in ("cos", "cosgram"):
                prediction = self.hidden_to_item_emb(selected.float()).float()
                if prediction.shape != target.shape:
                    raise ValueError("Alignment prediction and target dimensions differ")
                cos_loss = 1 - F.cosine_similarity(prediction, target, dim=-1).mean()
                align_loss = (self.cos_loss_weight * cos_loss + (1 - self.cos_loss_weight) * gram_loss
                              if self.align_loss_type == "cosgram" else cos_loss)
        return LCRecOutput(per_example.mean(), per_example, align_loss)

    @torch.no_grad()
    def generate(self, batch, num_beams=20, constrain_sid=True):
        prompt_width = batch["input_ids"].shape[1]
        layer_tokens = self.sid_token_ids.tolist()

        def allowed_tokens(batch_id, sequence):
            prefix = tuple(sequence[prompt_width:].tolist())
            if constrain_sid:
                # Completed/invalid zero-probability beam slots terminate safely.
                return self.prefix_tokens.get(prefix, [self.eos_token_id])
            return layer_tokens[len(prefix)] if len(prefix) < self.sem_id_dim else [self.eos_token_id]

        sequences = self.backbone.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            max_new_tokens=self.sem_id_dim, min_new_tokens=0,
            num_beams=num_beams, num_return_sequences=num_beams, do_sample=False,
            prefix_allowed_tokens_fn=allowed_tokens,
            eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
            forced_eos_token_id=None, forced_bos_token_id=None,
            length_penalty=0.0, use_cache=True, return_dict_in_generate=False,
        )
        sequences = sequences[:, prompt_width:prompt_width + self.sem_id_dim]
        sequences = F.pad(sequences, (0, self.sem_id_dim - sequences.shape[1]),
                          value=self.pad_token_id)
        return LCRecGeneration(sequences.view(-1, num_beams, self.sem_id_dim))

    def save_pretrained(self, path, tokenizer):
        """Save trainable adapters/embeddings and the tokenizer, not a second base LM."""
        import os
        self.backbone.save_pretrained(path)
        tokenizer.save_pretrained(path)
        if hasattr(self, "hidden_to_item_emb"):
            torch.save(self.hidden_to_item_emb.state_dict(), os.path.join(path, "alignment.pt"))
        if hasattr(self, "hidden_to_gram"):
            torch.save(self.hidden_to_gram.state_dict(), os.path.join(path, "gram_alignment.pt"))


def build_lcrec(base_model, tokenizer, sid_token_ids, catalogue_sids, mixed_precision="bf16",
                lora_r=8, lora_alpha=32, lora_dropout=0.05,
                lora_target_modules=("q_proj", "v_proj", "o_proj", "up_proj", "down_proj"),
                gradient_checkpointing=True, align_dim=None, align_source="last_inp",
                align_loss_type="cos", cos_loss_weight=0.5):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float32,
    )
    if backbone.config.model_type != "qwen3":
        raise ValueError("The LCRec backbone currently requires a Qwen3 causal LM")
    backbone.resize_token_embeddings(len(tokenizer))
    backbone.config.pad_token_id = tokenizer.pad_token_id
    backbone.config.eos_token_id = tokenizer.eos_token_id
    backbone.config.use_cache = False
    backbone = get_peft_model(backbone, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules), modules_to_save=["embed_tokens", "lm_head"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    ))
    if gradient_checkpointing:
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return LCRec(
        backbone, sid_token_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
        catalogue_sids, align_dim=align_dim, align_source=align_source,
        align_loss_type=align_loss_type, cos_loss_weight=cos_loss_weight,
    )
