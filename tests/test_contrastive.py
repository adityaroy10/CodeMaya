"""Tests for the contrastive-alignment primitives (pure torch)."""
import torch

from codemaya.training.contrastive_align import info_nce_loss, reinforce_loss


def test_info_nce_low_when_pairs_aligned():
    # matched text/image embeddings that are identical per row -> strong diagonal
    emb = torch.eye(4)
    aligned = info_nce_loss(emb, emb.clone(), temperature=0.07)
    # misaligned: shuffle images so the diagonal is no longer the match
    shuffled = emb[[1, 2, 3, 0]]
    misaligned = info_nce_loss(emb, shuffled, temperature=0.07)
    assert aligned < misaligned
    assert aligned >= 0.0


def test_reinforce_sign_and_detachment():
    logprob = torch.tensor([-2.0, -1.0], requires_grad=True)
    reward = torch.tensor([1.0, 0.0])
    loss = reinforce_loss(logprob, reward, baseline=0.5)
    assert loss.requires_grad and loss.dim() == 0
    loss.backward()
    # advantage = [0.5, -0.5]; loss = -mean(adv*logprob); dL/dlogprob = -adv/N
    assert torch.allclose(logprob.grad, torch.tensor([-0.25, 0.25]))


def test_reinforce_prefers_high_reward_actions():
    # raising logprob of an above-baseline action should lower the loss
    lp = torch.tensor([-1.0])
    r = torch.tensor([1.0])
    base = 0.0
    loss_lo = reinforce_loss(lp, r, base)
    loss_hi = reinforce_loss(lp + 0.5, r, base)  # more probable good action
    assert loss_hi < loss_lo
