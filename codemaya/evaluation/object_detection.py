"""Evaluation — prompt-guided object detection (paper 4.9).

A ViT image classifier is trained to recognise the rendered object class
(cube / sphere / …). At eval time it verifies that a render matches the object
the prompt asked for: "a cube on a chair" -> the render should be classified
`cube`. The paper reports ~0.96 classifier accuracy over 30 objects × 5 views.

Pipeline: render primitives -> ViT CLS features (frozen) -> a small linear probe
trained on top -> accuracy / precision / recall. Uses a MockEncoder + tiny probe
in smoke mode so it runs offline; the real path uses `eval.vit_model`.
"""
from __future__ import annotations

from pathlib import Path

from codemaya.utils.io import ensure_dir, write_json
from codemaya.utils.logging import get_logger

log = get_logger("eval.detect")

DEFAULT_CLASSES = ["cube", "sphere"]


class LinearProbe:
    """Multinomial logistic-regression head on frozen features (torch)."""

    def __init__(self, n_classes: int, epochs: int = 200, lr: float = 0.05):
        self.n_classes, self.epochs, self.lr = n_classes, epochs, lr
        self.W = None

    def fit(self, X, y):
        import torch
        X = X.float()
        self.W = torch.zeros(X.size(1), self.n_classes, requires_grad=True)
        b = torch.zeros(self.n_classes, requires_grad=True)
        opt = torch.optim.Adam([self.W, b], lr=self.lr)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(X @ self.W + b, y)
            loss.backward()
            opt.step()
        self._b = b.detach()
        self.W = self.W.detach()
        return self

    def predict(self, X):
        import torch
        return torch.argmax(X.float() @ self.W + self._b, dim=1)


def render_primitive_dataset(cfg, classes, n_per_class, views):
    """Render `n_per_class` instances of each class from `views` viewpoints."""
    import random

    from codemaya.rendering.render import build_mesh, render_views

    rng = random.Random(int(cfg.seed))
    images, labels = [], []
    for ci, kind in enumerate(classes):
        for _ in range(n_per_class):
            size = rng.uniform(0.8, 2.0)
            mesh = build_mesh({"kind": kind, "radius": size, "width": size, "height": size})
            for img in render_views(mesh, int(cfg.render.image_size), views):
                images.append(img)
                labels.append(ci)
    return images, labels


def _features(encoder, images):
    return encoder.encode_image(images)


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    import torch
    y_true, y_pred = torch.as_tensor(y_true), torch.as_tensor(y_pred)
    acc = float((y_true == y_pred).float().mean())
    per_class = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3)}
    return {"accuracy": round(acc, 3), "per_class": per_class}


def run(cfg, args) -> None:
    classes = DEFAULT_CLASSES
    if args.smoke:
        from codemaya.evaluation.embeddings import MockEncoder
        encoder = MockEncoder()
        n_per, views = 3, 2
    else:
        from codemaya.evaluation.embeddings import ViTEncoder
        encoder = ViTEncoder(cfg.eval.vit_model)
        n_per, views = 15, int(cfg.render.viewpoints)

    images, labels = render_primitive_dataset(cfg, classes, n_per, views)
    import torch
    X = _features(encoder, images)
    y = torch.tensor(labels)

    # simple split: alternate indices into train/test
    idx = torch.arange(len(y))
    train_m, test_m = idx % 2 == 0, idx % 2 == 1
    probe = LinearProbe(len(classes)).fit(X[train_m], y[train_m])
    pred = probe.predict(X[test_m])
    metrics = classification_metrics(y[test_m], pred, len(classes))
    metrics["classes"] = classes
    metrics["n_images"] = len(images)

    write_json(Path(ensure_dir(cfg.paths.results)) / "detection.json", metrics)
    log.info("prompt-guided detection accuracy=%.3f over %d imgs (%s)",
             metrics["accuracy"], len(images), classes)
