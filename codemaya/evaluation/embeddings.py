"""Shared vision/text encoders for evaluation: CLIP and DINOv2.

- **CLIP (ViT-B/32)** — joint text+image space; used for prompt<->render
  semantic similarity (paper 4.8) and as the stage-2 reward.
- **DINOv2** — self-supervised ViT image features; used for render<->ground-truth
  visual similarity (paper 4.3).

`MockEncoder` returns deterministic embeddings so metric aggregation is testable
offline; the real encoders download weights from the HF Hub on first use.
"""
from __future__ import annotations


def cosine_sim(a, b):
    """Row-wise cosine between two [N, D] tensors -> [N]."""
    import torch
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(-1)


def cosine_matrix(a, b):
    """Full [Na, Nb] cosine similarity matrix."""
    import torch
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return a @ b.t()


class ClipEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._torch = torch

    def encode_text(self, texts):
        torch = self._torch
        with torch.no_grad():
            inp = self.processor(text=list(texts), return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            return self.model.get_text_features(**inp).cpu()

    def encode_image(self, images):
        torch = self._torch
        with torch.no_grad():
            inp = self.processor(images=list(images), return_tensors="pt").to(self.device)
            return self.model.get_image_features(**inp).cpu()


class HFImageEncoder:
    """Generic HF ViT-style image encoder (DINOv2, ViT, …) -> CLS embedding."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self._torch = torch

    def encode_image(self, images):
        torch = self._torch
        with torch.no_grad():
            inp = self.processor(images=list(images), return_tensors="pt").to(self.device)
            out = self.model(**inp)
            # CLS token if available, else mean-pool the patch tokens
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output.cpu()
            return out.last_hidden_state[:, 0].cpu()


# DINOv2 (visual similarity) and ViT (detection / siamese) share the pattern.
class DinoEncoder(HFImageEncoder):
    pass


class ViTEncoder(HFImageEncoder):
    pass


class MockEncoder:
    """Deterministic random encoder (offline smoke/tests). dim defaults to 16."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _emb(self, items):
        import torch
        embs = []
        for x in items:
            g = torch.Generator().manual_seed(hash(str(x)) % (2**31))
            embs.append(torch.rand(self.dim, generator=g))
        return torch.stack(embs)

    def encode_text(self, texts):
        return self._emb(texts)

    def encode_image(self, images):
        # key on actual pixel content so distinct renders get distinct embeddings
        keys = []
        for i, im in enumerate(images):
            try:
                keys.append(hash(im.tobytes()))
            except Exception:  # noqa: BLE001 - non-PIL fallback
                keys.append(i)
        return self._emb(keys)
