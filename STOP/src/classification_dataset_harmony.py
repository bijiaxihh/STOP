import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ClassificationDatasetHarmony(Dataset):
    """Dataset for preprocessed Harmony prefix records."""

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=None,
        use_hard_labels=False,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.use_hard_labels = use_hard_labels

        self.data = []
        with open(data_path, "r", encoding="utf-8") as file_object:
            for line in file_object:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prefix_token_ids = self._resolve_prefix_token_ids(item, idx)
        if self.max_length is not None and len(prefix_token_ids) > self.max_length:
            raise ValueError(
                f"[DataFormatError] Item {idx} in '{self.data_path}' has "
                f"{len(prefix_token_ids)} prefix tokens, exceeding max_length={self.max_length}."
            )

        label_prob = self._resolve_label_prob(item, idx)
        return {
            "prefix_token_ids": prefix_token_ids,
            "label_prob": label_prob,
        }

    def collate_fn(self, batch):
        prefix_tensors = [
            torch.tensor(item["prefix_token_ids"], dtype=torch.long)
            for item in batch
        ]
        prefix_lengths = torch.tensor(
            [tensor.numel() for tensor in prefix_tensors],
            dtype=torch.long,
        )
        pad_token_id = self.tokenizer.pad_token_id
        prefix_ids = pad_sequence(
            prefix_tensors,
            batch_first=True,
            padding_value=pad_token_id,
            padding_side="left",
        )
        batch_size, max_seq_len = prefix_ids.shape
        position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        valid_start = (max_seq_len - prefix_lengths).unsqueeze(1)
        prefix_attention_mask = (position_ids >= valid_start).long()

        soft_labels = []
        for item in batch:
            good_prob = float(item["label_prob"])
            bad_prob = 1.0 - good_prob
            soft_labels.append([bad_prob, good_prob])

        labels = torch.tensor(soft_labels, dtype=torch.float32)
        return {
            "prefix_ids": prefix_ids,
            "prefix_attention_mask": prefix_attention_mask,
            "labels": labels,
        }

    def _resolve_prefix_token_ids(self, item, idx):
        value = item.get("prefix_token_ids")
        if isinstance(value, list) and value:
            return [int(token_id) for token_id in value]

        raise ValueError(
            f"[DataFormatError] Item {idx} in '{self.data_path}' does not contain "
            "a usable 'prefix_token_ids' field."
        )

    def _resolve_label_prob(self, item, idx):
        # Hard-label training is intentionally disabled for the current soft-label run.
        # Keep this branch commented for later reuse if you need to switch back.
        #
        # if self.use_hard_labels:
        #     for key in ("Hard_label", "hard_label", "label", "class_label"):
        #         if key in item:
        #             return float(item[key])
        #     raise ValueError(
        #         f"[DataFormatError] Item {idx} in '{self.data_path}' is missing a hard label."
        #     )

        for key in (
            "Soft_label",
            "soft_label",
            "good_probability",
            "label_prob",
            "score",
        ):
            if key in item:
                return float(item[key])

        # Keep the old hard-label fallback commented out so Soft_label is the only active source.
        #
        # if "Hard_label" in item:
        #     return float(item["Hard_label"])
        # if "class_label" in item:
        #     return float(item["class_label"])

        raise ValueError(
            f"[DataFormatError] Item {idx} in '{self.data_path}' is missing a soft label."
        )
