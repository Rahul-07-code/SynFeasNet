import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

CHEMBERTA_MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_MAX_LENGTH = 320


class SMILESTokenizer:
    def __init__(self, max_length: int = DEFAULT_MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_ID)
        self.max_length = max_length
        print(
            f"  SMILESTokenizer loaded | "
            f"vocab_size={self.tokenizer.vocab_size} | "
            f"max_length={max_length}"
        )

    def __call__(self, smiles, device=None, dynamic_truncation: bool = True) -> dict:
        if isinstance(smiles, str):
            smiles = [smiles]

        # Dynamic truncation: use minimum of self.max_length or the actual required length
        # to save memory during batch processing.
        curr_max = self.max_length
        if dynamic_truncation and isinstance(smiles, list) and len(smiles) > 0:
            # Heuristic: max length of this specific batch
            actual_max = max([len(s) for s in smiles]) 
            curr_max = min(self.max_length, actual_max + 2) # small buffer

        encoded = self.tokenizer(
            smiles,
            max_length=curr_max,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        return encoded


class ChemBERTaBranch(nn.Module):
    CLS_DIM = 768
    OUTPUT_DIM = 256

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        print(f"  Loading ChemBERTa: {CHEMBERTA_MODEL_ID}")
        base_model = AutoModel.from_pretrained(
            CHEMBERTA_MODEL_ID,
            use_safetensors=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            bias="none",
        )

        self.encoder = get_peft_model(base_model, lora_config)

        trainable, total = self._count_parameters()
        print(
            f"  LoRA applied | trainable: {trainable:,} / "
            f"total: {total:,} ({100 * trainable / total:.1f}%)"
        )

        self.projection = nn.Sequential(
            nn.Linear(self.CLS_DIM, self.OUTPUT_DIM),
            nn.ReLU(),
        )
        self._init_projection()

    def _init_projection(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _count_parameters(self):
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        return trainable, total

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)
