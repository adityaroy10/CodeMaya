"""Stage 2 — CLIP contrastive alignment loop.

The paper renders each generated MEL script and "retrains the model to align the
embeddings of the prompt and the corresponding rendered output" using a
contrastive loss over frozen CLIP encoders — so the model learns to emit scripts
whose *renders* match the prompt, not just scripts that parse.

Gradient reality: rendering (MEL -> image, via Maya) is non-differentiable, so
the CLIP alignment signal cannot flow back through the render by autograd. We
therefore optimize it the standard way for a non-differentiable reward over an
autoregressive generator: **policy gradient (REINFORCE)**. Per step:

    1. generate a MEL script for each prompt (current LoRA policy)
    2. render it to an image (pluggable renderer)
    3. reward_i = CLIP cosine(prompt_i, render_i)      [frozen CLIP]
    4. loss = -mean( (reward_i - baseline) * logprob_theta(gen_i | prompt_i) )
    5. step the LoRA params only (base model + CLIP frozen)

`info_nce_loss` is the symmetric CLIP-style contrastive loss over a batch and is
kept as a pure, unit-tested primitive (it also scores batch alignment quality).

`--smoke` uses a tiny model + mock renderer + mock reward and computes ONE loss
value without stepping the optimizer (no real training).
"""
from __future__ import annotations

from pathlib import Path

from codemaya.prompts import format_prompt
from codemaya.utils.io import read_jsonl
from codemaya.utils.logging import get_logger

log = get_logger("contrastive")

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ------------------------------------------------------------- core primitive -
def info_nce_loss(text_emb, image_emb, temperature: float = 0.07):
    """Symmetric InfoNCE (CLIP loss) over a batch of matched (text, image) pairs.

    text_emb, image_emb: [B, D] tensors. Positives lie on the diagonal.
    Returns a scalar loss (lower = better alignment of matched pairs).
    """
    import torch
    import torch.nn.functional as F

    t = F.normalize(text_emb, dim=-1)
    v = F.normalize(image_emb, dim=-1)
    logits = (t @ v.t()) / temperature          # [B, B]
    targets = torch.arange(t.size(0), device=t.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


# ---------------------------------------------------------------- reward side -
class MockRewardModel:
    """Deterministic pseudo-random rewards in [0, 1] — offline smoke/testing."""

    def clip_reward(self, prompts, images):
        import torch
        g = torch.Generator().manual_seed(len(prompts))
        return torch.rand(len(prompts), generator=g)


class ClipRewardModel:
    """Frozen CLIP: per-sample cosine similarity between prompt and render."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def clip_reward(self, prompts, images):
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(text=list(prompts), images=list(images),
                                    return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = self.model(**inputs)
            t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            v = torch.nn.functional.normalize(out.image_embeds, dim=-1)
            return (t * v).sum(-1).cpu()        # [B] cosine of matched pairs


# ------------------------------------------------------------ policy gradient -
def sequence_logprob(model, tokenizer, prompt_texts, gen_texts, device="cpu"):
    """Sum log-prob of gen tokens under `model`, conditioned on the prompt.

    Teacher-forced pass over [prompt + gen]; only the gen-token positions count.
    Returns a [B] tensor with grad w.r.t. the model's (LoRA) params.
    """
    import torch

    logprobs = []
    for prompt, gen in zip(prompt_texts, gen_texts):
        p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        g_ids = tokenizer(gen, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if g_ids.size(1) == 0:                                  # empty generation
            logprobs.append(torch.zeros((), device=device))
            continue
        full = torch.cat([p_ids, g_ids], dim=1)
        logits = model(full).logits                            # [1, T, V]
        # predict token t from position t-1; score only the gen span
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = full[:, 1:]
        tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        gen_logp = tok_logp[:, p_ids.size(1) - 1:]             # gen positions only
        logprobs.append(gen_logp.sum())
    return torch.stack(logprobs)


def reinforce_loss(logprob, reward, baseline):
    """REINFORCE: -mean((reward - baseline) * logprob). `reward` detached."""
    advantage = (reward - baseline).detach()
    return -(advantage * logprob).mean()


# --------------------------------------------------------------------- loop --
class ContrastiveAligner:
    def __init__(self, model, tokenizer, reward_model, renderer, cfg, device="cpu"):
        self.model, self.tokenizer = model, tokenizer
        self.reward_model, self.renderer = reward_model, renderer
        self.cfg, self.device = cfg, device
        self._baseline = 0.0

    def generate(self, prompts, max_new_tokens=64):
        import torch

        gens = []
        for p in prompts:
            ids = self.tokenizer(format_prompt(p), return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                                          top_p=0.95, pad_token_id=self.tokenizer.pad_token_id)
            gens.append(self.tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True))
        return gens

    def compute_loss(self, prompts):
        gens = self.generate(prompts)
        images = [self.renderer(mel) for mel in gens]
        reward = self.reward_model.clip_reward(prompts, images)
        logprob = sequence_logprob(self.model, self.tokenizer, prompts, gens, self.device)
        loss = reinforce_loss(logprob, reward, self._baseline)
        # EMA baseline reduces gradient variance
        self._baseline = 0.9 * self._baseline + 0.1 * float(reward.mean())
        return loss, float(reward.mean())


def _mock_renderer(mel: str):
    from PIL import Image
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


def run(cfg, args) -> None:
    import torch

    base = TINY_MODEL if args.smoke else cfg.model.base
    prompts = _load_prompts(cfg, limit=cfg.contrastive.batch_size if args.smoke else None)

    from codemaya.inference.generate import load_policy   # base + SFT LoRA adapter
    tokenizer, model = load_policy(cfg, base=base, smoke=args.smoke)

    if args.smoke:
        reward_model, renderer = MockRewardModel(), _mock_renderer
    else:
        reward_model = ClipRewardModel(cfg.contrastive.clip_model)
        from codemaya.rendering.render import make_renderer
        renderer = make_renderer(cfg)

    aligner = ContrastiveAligner(model, tokenizer, reward_model, renderer, cfg)

    if args.smoke:
        loss, mean_r = aligner.compute_loss(prompts[: int(cfg.contrastive.batch_size)])
        log.info("smoke: one contrastive step computed loss=%.4f mean_reward=%.4f (no optimizer step)",
                 float(loss.detach()), mean_r)
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(cfg.contrastive.lr))
    bs = int(cfg.contrastive.batch_size)
    for step in range(int(cfg.contrastive.steps)):
        batch = prompts[(step * bs) % len(prompts):][:bs] or prompts[:bs]
        opt.zero_grad()
        loss, mean_r = aligner.compute_loss(batch)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            log.info("step %d | loss=%.4f | mean_reward=%.4f", step, float(loss), mean_r)
    out = Path(cfg.paths.checkpoints) / "contrastive_lora"
    model.save_pretrained(str(out))
    log.info("saved aligned adapter -> %s", out)


def _load_prompts(cfg, limit=None) -> list[str]:
    path = Path(cfg.paths.dataset) / "train.jsonl"
    rows = list(read_jsonl(path)) if path.exists() else [{"prompt": "Create a sphere in Maya."}]
    prompts = [r["prompt"] for r in rows]
    return prompts[:limit] if limit else prompts
