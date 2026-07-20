"""Tests for syntax validity and classification metrics."""
import torch

from codemaya.evaluation.object_detection import LinearProbe, classification_metrics
from codemaya.evaluation.siamese_similarity import pairwise_similarity_table
from codemaya.evaluation.syntax_validity import validate_mel_heuristic


def test_syntax_accepts_valid_mel():
    assert validate_mel_heuristic('polyCube -w 2 -name "box";')["valid"]
    assert validate_mel_heuristic("sphere -r 1;")["valid"]
    assert validate_mel_heuristic("string $s = `polySphere -r 2`;")["valid"]


def test_syntax_rejects_broken_mel():
    assert not validate_mel_heuristic("")["valid"]
    assert not validate_mel_heuristic('polyCube -name "box;')["valid"]   # open quote
    assert not validate_mel_heuristic("sphere -r ((1;")["valid"]         # open paren


def test_classification_metrics_perfect():
    y = [0, 0, 1, 1]
    m = classification_metrics(y, y, n_classes=2)
    assert m["accuracy"] == 1.0
    assert m["per_class"][0]["precision"] == 1.0 and m["per_class"][1]["recall"] == 1.0


def test_linear_probe_separates_toy_data():
    # two linearly separable clusters
    g = torch.Generator().manual_seed(0)
    a = torch.randn(30, 4, generator=g) + torch.tensor([3.0, 0, 0, 0])
    b = torch.randn(30, 4, generator=g) + torch.tensor([-3.0, 0, 0, 0])
    X = torch.cat([a, b]); y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    probe = LinearProbe(2, epochs=300).fit(X, y)
    acc = (probe.predict(X) == y).float().mean()
    assert acc > 0.95


def test_siamese_table_within_higher_than_cross():
    # construct embeddings: class A near [1,0], class B near [0,1]
    a = torch.tensor([[1.0, 0.05], [0.98, 0.0], [1.0, 0.1]])
    b = torch.tensor([[0.0, 1.0], [0.05, 0.99], [0.0, 0.97]])
    table = pairwise_similarity_table({"cube": a, "sphere": b})
    assert table["cube-vs-cube"] > table["cube-vs-sphere"]
    assert table["sphere-vs-sphere"] > table["cube-vs-sphere"]
