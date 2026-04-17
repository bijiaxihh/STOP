import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class HarmonyTwoStageClassifier(nn.Module):
    """Two-stage prefix scorer for raw Harmony-format gpt-oss traces."""

    def __init__(
        self,
        model_path: str,
        lora_config: LoraConfig,
        num_labels: int = 2,
        num_assess_tokens: int = 4,
        assess_token: str = "[ASSESS]",
        torch_dtype=None,
        local_files_only=True,
        device_map=None,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.use_gradient_checkpointing = False

        model_load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if torch_dtype is not None:
            model_load_kwargs["dtype"] = torch_dtype
        if device_map is not None:
            model_load_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_gradient_checkpointing:
            print(
                "HarmonyTwoStageClassifier disables gradient checkpointing because "
                "cache-based assess scoring requires past_key_values during training."
            )
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if hasattr(self.model, "_require_grads_hook") and hasattr(
            self.model, "disable_input_require_grads"
        ):
            self.model.disable_input_require_grads()

        self.assess_token = assess_token
        if self.assess_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.assess_token])
        self.special_token_id = self.tokenizer.convert_tokens_to_ids(self.assess_token)

        input_embeddings = self.model.get_input_embeddings()
        num_embeddings = getattr(input_embeddings, "num_embeddings", None)
        if num_embeddings is None and hasattr(input_embeddings, "weight"):
            num_embeddings = input_embeddings.weight.shape[0]
        if num_embeddings is None:
            raise RuntimeError("Unable to determine base model vocabulary size.")
        if self.special_token_id >= num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
            input_embeddings = self.model.get_input_embeddings()
        self.assess_token_embedding = nn.Parameter(
            input_embeddings.weight[self.special_token_id].detach().clone()
        )

        self.model = get_peft_model(self.model, lora_config, adapter_name="classifier")
        self._cast_lora_parameters(torch_dtype)
        self.num_assess_tokens = num_assess_tokens
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is None:
            model_dtype = next(self.model.parameters()).dtype
        self.classifier_head = nn.Linear(
            self.model.config.hidden_size,
            num_labels,
        ).to(dtype=model_dtype)
        self.loss_fct = nn.CrossEntropyLoss()

        print("LoRA adapter 'classifier' added to the model.")
        self.model.print_trainable_parameters()

    def _cast_lora_parameters(self, target_dtype) -> None:
        if target_dtype is None:
            return

        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if "lora_" not in name:
                    continue
                if parameter.dtype == target_dtype:
                    continue
                parameter.data = parameter.data.to(dtype=target_dtype)

    def load_lora_parameters(self, load_path, adapter_name):
        lora_params = torch.load(load_path, map_location=self.device)
        to_load = {name.replace("default", adapter_name): value for name, value in lora_params.items()}
        self.load_state_dict(to_load, strict=False)
        print(f"Loaded LoRA parameters from {load_path} into adapter '{adapter_name}'")

    @property
    def device(self):
        return self.classifier_head.weight.device

    def forward(self, prefix_ids, prefix_attention_mask, labels=None):
        batch_size, seq_length = prefix_ids.shape

        padding_len = seq_length - prefix_attention_mask.sum(dim=1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=prefix_ids.device,
        ).unsqueeze(0).expand(batch_size, seq_length)
        prefix_position_ids = position_ids - padding_len.unsqueeze(1)
        prefix_position_ids = prefix_position_ids.masked_fill(prefix_position_ids < 0, 0)

        # Stage 1: encode the candidate prefix with the base model only.
        # LoRA is disabled here on purpose so the scorer is trained against the
        # same prefix distribution used by the base-model-only inference branch.
        with torch.no_grad():
            with self.model.disable_adapter():
                prefix_outputs = self.model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_attention_mask,
                    position_ids=prefix_position_ids,
                    use_cache=True,
                )
        prefix_kv_cache = prefix_outputs.past_key_values
        if prefix_kv_cache is None:
            raise RuntimeError(
                "Prefix KV cache is missing. This scorer requires use_cache=True "
                "to survive the stage-1 prefix pass."
            )

        # Stage 2: append learned assess embeddings and score the frozen base
        # prefix using the classifier LoRA branch plus the classifier head.
        self.model.set_adapter("classifier")
        assess_inputs_embeds = self.assess_token_embedding.to(prefix_ids.device)
        assess_inputs_embeds = assess_inputs_embeds.view(1, 1, -1).expand(
            batch_size,
            self.num_assess_tokens,
            -1,
        )
        assess_attention_mask = torch.ones(
            (batch_size, self.num_assess_tokens),
            dtype=torch.long,
            device=prefix_ids.device,
        )
        combined_attention_mask = torch.cat(
            [prefix_attention_mask, assess_attention_mask],
            dim=1,
        )

        valid_prefix_len = prefix_attention_mask.sum(dim=1).unsqueeze(-1)
        assess_pos_offset = torch.arange(
            self.num_assess_tokens,
            dtype=torch.long,
            device=prefix_ids.device,
        ).unsqueeze(0)
        assess_position_ids = valid_prefix_len + assess_pos_offset

        assess_outputs = self.model(
            inputs_embeds=assess_inputs_embeds,
            past_key_values=prefix_kv_cache,
            attention_mask=combined_attention_mask,
            position_ids=assess_position_ids,
            output_hidden_states=True,
        )

        assess_hidden_states = assess_outputs.hidden_states[-1]
        last_token_hidden_state = assess_hidden_states[:, -1, :].to(
            self.classifier_head.weight.dtype
        )
        classification_logits = self.classifier_head(last_token_hidden_state)

        output = {"logits": classification_logits}
        if labels is not None:
            output["loss"] = self.loss_fct(classification_logits, labels)
        return output
