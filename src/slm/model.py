"""
Integrated SLM (Small Language Model) wrapper.
PhoBERT-based binary classifier for fake news detection (Vietnamese).

Supports two backends:
- "hf": HuggingFace Transformers (default)
- "vllm": vLLM for faster inference (requires vllm package) - not fully supported for classification.
"""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.config import MODEL_PATH, SLM_BACKEND
from src.utils import preprocess_text
from src.slm.dataset import FakeNewsDataset


class IntegratedSLM:
    """
    Wrapper for PhoBERT-based SLM with inference and fine-tuning capabilities.
    
    Args:
        model_path: Path to pre-trained model checkpoint (or HuggingFace model name).
        backend: "hf" (HuggingFace) or "vllm"
    """

    def __init__(self, model_path: str = MODEL_PATH, backend: str = None):
        self.backend = backend or SLM_BACKEND
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_model_path = None
        
        if not os.path.exists(model_path):
            print("Saved model not found. Using pre-trained PhoBERT base.")
            model_path = "vinai/phobert-base"
        else:
            print(f"Loading SLM from {model_path}")
        
        self._loaded_model_path = model_path

        if self.backend == "vllm":
            self._init_vllm(model_path)
        else:
            self._init_hf(model_path)

    # ================================================================
    # HuggingFace Backend
    # ================================================================
    def _init_hf(self, model_path: str):
        self.tokenizer, self.model = self._load_phobert_components(
            model_path=model_path,
            eval_mode=True,
        )
        print(f"SLM loaded (HF backend with PhoBERT) on {self.device}")

    def _load_phobert_components(self, model_path: str, eval_mode: bool = True):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
        )
        model.to(self.device)
        if eval_mode:
            model.eval()
        else:
            model.train()
        return tokenizer, model

    # ================================================================
    # Inference
    # ================================================================
    def _inference_hf(self, text: str) -> tuple:
        clean_text = preprocess_text(text)
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        with torch.no_grad():
            outputs = self.model(
                inputs["input_ids"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )
            probs = F.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.item(), conf.item(), probs[0].cpu().numpy()

    def _inference_hf_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        clean_texts = [preprocess_text(t) for t in texts]
        results = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = self.model(
                    inputs["input_ids"].to(self.device),
                    inputs["attention_mask"].to(self.device),
                )
                probs = F.softmax(outputs.logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                for j in range(len(batch_texts)):
                    results.append((pred[j].item(), conf[j].item(), probs[j].cpu().numpy()))
        return results

    def inference(self, text: str) -> tuple:
        if self.backend == "vllm":
            return self._inference_vllm(text)
        return self._inference_hf(text)

    def inference_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        return self._inference_hf_batch(texts, batch_size)

    # ================================================================
    # Full fine‑tuning (cả backbone + head)
    # ================================================================
    def finetune_full(
        self,
        train_texts: list[str],
        train_labels: list[int],
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_path: str | None = None,
    ) -> dict:
        """Full fine‑tune toàn bộ mô hình (PhoBERT backbone + classification head)."""
        if len(train_texts) != len(train_labels):
            raise ValueError("train_texts và train_labels phải cùng số lượng")
        if len(train_texts) == 0:
            return {"trained": False, "reason": "no_train_data"}

        # Load model từ pretrained (hoặc checkpoint hiện tại)
        self.tokenizer, self.model = self._load_phobert_components(
            model_path=self._loaded_model_path,
            eval_mode=False,      # train mode
        )
        self.model.train()

        dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer, max_len=128)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_steps = len(loader) * epochs
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        label_counts = torch.tensor(
            [
                sum(1 for l in train_labels if l == 0),
                sum(1 for l in train_labels if l == 1),
            ],
            dtype=torch.float,
        )
        class_weights = (label_counts.sum() / (2 * label_counts.clamp(min=1))).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        history = {"train_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_t = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)
            print(f"[Full FT] Epoch {epoch+1}/{epochs} | Loss {avg_loss:.4f}")

        self.model.eval()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

        return {
            "trained": True,
            "samples": len(train_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_loss_history": history["train_loss"],
            "save_path": save_path,
        }

    # ================================================================
    # Head‑only fine‑tuning (chỉ train classification head) – dùng trong MRCD
    # ================================================================
    def _freeze_backbone_train_head_only(self):
        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            raise AttributeError("Model does not expose 'classifier' head")

    def _set_head_train_mode(self):
        self.model.eval()
        if hasattr(self.model, 'classifier'):
            self.model.classifier.train()

    def finetune_on_clean(
        self,
        clean_samples: list,
        epochs: int = 2,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> dict:
        """
        Chỉ train classification head trên tập D_clean (dùng trong MRCD).
        Backbone PhoBERT được đóng băng hoàn toàn.
        """
        valid_samples = [
            s for s in clean_samples
            if s.get("text") is not None and s.get("label") in [0, 1]
        ]
        if not valid_samples:
            return {"trained": False, "reason": "no_valid_samples"}

        texts = [preprocess_text(s["text"]) for s in valid_samples]
        labels = [int(s["label"]) for s in valid_samples]

        # Đóng băng backbone, chỉ để head trainable
        self._freeze_backbone_train_head_only()

        dataset = FakeNewsDataset(texts, labels, self.tokenizer, max_len=128)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        label_counts = torch.tensor(
            [
                sum(1 for label in labels if label == 0),
                sum(1 for label in labels if label == 1),
            ],
            dtype=torch.float,
        )
        class_weights = (label_counts.sum() / (2 * label_counts.clamp(min=1))).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        total_loss = 0.0
        total_steps = 0

        for _ in range(epochs):
            self._set_head_train_mode()   # backbone eval, head train
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_t = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_steps += 1

        self.model.eval()
        avg_loss = total_loss / max(1, total_steps)
        return {
            "trained": True,
            "samples": len(valid_samples),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "avg_loss": avg_loss,
        }

    # ================================================================
    # Legacy methods (giữ để tương thích)
    # ================================================================
    def finetune(
        self,
        train_texts: list[str],
        train_labels: list[int],
        model_init: str = "vinai/phobert-base",
        epochs: int = 4,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_path: str | None = None,
    ) -> dict:
        """Head‑only fine‑tune (giống finetune_on_clean nhưng dùng cho initial training)."""
        # Giống như code cũ của bạn
        if len(train_texts) != len(train_labels):
            raise ValueError("train_texts và train_labels phải cùng số lượng")
        if len(train_texts) == 0:
            return {"trained": False, "reason": "no_train_data"}

        if self._loaded_model_path != model_init:
            self.tokenizer, self.model = self._load_phobert_components(
                model_path=model_init,
                eval_mode=True,
            )
            self._loaded_model_path = model_init

        self._freeze_backbone_train_head_only()

        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer, max_len=128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        total_steps = len(train_loader) * epochs
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        label_counts = torch.tensor(
            [
                sum(1 for l in train_labels if l == 0),
                sum(1 for l in train_labels if l == 1),
            ],
            dtype=torch.float,
        )
        class_weights = (label_counts.sum() / (2 * label_counts.clamp(min=1))).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        history = {"train_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            self._set_head_train_mode()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_t = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())

            avg_train_loss = epoch_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)

        self.model.eval()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

        result = {
            "trained": True,
            "samples": len(train_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_loss_history": history["train_loss"],
        }
        if save_path:
            result["save_path"] = save_path
        return result

    def fnetune(self, train_texts, train_labels, model_init="vinai/phobert-base", 
                epochs=4, batch_size=32, lr=1e-3, weight_decay=1e-4, 
                warmup_ratio=0.1, max_grad_norm=1.0, save_path=None):
        return self.finetune(
            train_texts=train_texts, train_labels=train_labels,
            model_init=model_init, epochs=epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm, save_path=save_path
        )

    # vLLM backend (giữ nguyên)
    def _init_vllm(self, model_path: str):
        try:
            from vllm import LLM as VLLM_LLM
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        self.tokenizer, self.model = self._load_phobert_components(
            model_path=model_path,
            eval_mode=True,
        )
        self._vllm_model_path = model_path
        print(f"SLM loaded (vLLM backend fallback to HF) on {self.device}")

    def _inference_vllm(self, text: str) -> tuple:
        return self._inference_hf(text)