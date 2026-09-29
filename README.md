# GRIT: Process-Guided Gradient Projection and Trust-Region Anchoring

Training workflow for **GRIT** — a forgetting-resistant RL fine-tuning method that combines
process-reward-shaped credit assignment, process-gated soft null-space projection, and
competence-gated trust-region correction (Le Van Dong, Nguyen Hoang Nam, Nguyen Quy Duong, 2026).

This README describes the end-to-end training pipeline (Algorithm 1 in the paper), the
artifacts each stage produces/consumes, and the knobs you'll actually tune.

---

## 1. Overview

GRIT applies three mechanisms inside a single RL training step:

1. **Process-shaped credit** — a frozen process reward model (PRM) reshapes the outcome
   advantage into per-token credit. Provably leaves the policy gradient and optimal-policy
   set unchanged (Prop. 3); it only reduces gradient variance.
2. **Process-gated soft projection** — task gradients on protected weight matrices are passed
   through a relaxed null-space projector `P_κ`, where leakage `κ_t` per token is set by PRM
   step progress. Produces a **predictor** point.
3. **Competence-gated trust-region correction** — a monitoring batch checks the predictor
   against the frozen base policy; wherever divergence exceeds a PRM-competence-gated
   tolerance, a differentiable output-space projection pulls it back.

The three stages are unrolled into **one objective at θ** (Eq. 18) so a single backward pass
produces the full update.

---

## 2. Prerequisites

| Component | Purpose | Notes |
|---|---|---|
| `π_base` | Frozen base checkpoint | Source of both the null-space subspace and the trust-region anchor |
| `π_θ` | Trainable policy | Initialized from `π_base` |
| PRM `R_ψ` | Frozen process reward model | Must output `Φ_ψ(s) ∈ [0,1]`, with `Φ_ψ(s_T) = 0` enforced at terminal states |
| `D_task` | Task prompts + verifiable/learned reward | Drives `J_task` |
| `D_pres` | Preservation prompts (code, logic, multilingual, instruction-following, etc.) | Used both to build the projector and as the monitoring batch |

**PRM coverage requirement:** the PRM must be calibrated on `D_pres` domains, not just
`D_task`. Scoring preservation rollouts with a math-only PRM silently voids the guarantee
(§3.4 caution / §6 limitations).

---

## 3. One-time setup: build the null-space projector

Run once, before training starts:

1. Roll out `π_base` over `D_pres`.
2. For each protected weight matrix `W`, collect the input-activation matrix `K` (shape `d × N`, `N ≫ d`).
3. Compute the non-central covariance `K Kᵀ = U Λ Uᵀ` (avoids factorizing `K` directly).
4. Keep eigenvectors `Û` whose eigenvalues fall below threshold `5e-4` (paper default) → define
   `P = Û Ûᵀ`.

This `P` is **not refreshed** during training (see Limitations, §7). Periodic recomputation is
a known open cost, especially at higher `κ_max`.

---

## 4. Per-step training loop (Algorithm 1)

```
for step n = 1, 2, ...:
    # -- Task rollout --
    sample task batch from D_task
    roll out under π_θ; set π_old ← π_θ

    # -- Process-shaped credit (§3.2) --
    score rollouts with R_ψ
    ϱ_t = γ·Φ_ψ(s_{t+1}) − Φ_ψ(s_t)
    Â_{i,t} = Â_i + ϱ_{i,t}                      # Eq. 5

    # -- Process-gated soft projection (§3.3) --
    κ_t = κ_max · σ(ϱ_t / τ_κ)
    g_A = Σ_t Â_t · g_t
    g_κ = Σ_t κ_t · Â_t · g_t
    g   = P·g_A + (I − P)·g_κ                    # Eq. 10

    # -- Predictor (§3.4) --
    θ_tilde = θ + α·g                             # Eq. 11

    # -- Competence-gated correction --
    sample monitoring batch from D_pres
    roll out under π_θ_tilde AND π_base; score both with R_ψ
    for each token:
        m_t(q) = Φ_ψ(s_tilde_t) − Φ_ψ(s_base_t)   # competence margin, Eq. 4
        ε_t(q) = ε_min + (ε_max − ε_min)·σ(m_t/τ_ε)   # Eq. 13
        δ_t(q) = KL(π_θ_tilde(o_t) ‖ π_base(o_t))     # Eq. 12
        if δ_t(q) ≤ ε_t(q):
            η* = 0          # no correction needed
        else:
            solve η* by bracketing search (Lemma 1 guarantees monotonicity)
            form π_proj via geometric interpolation of logits (Eq. 15)

    v = ∇_θ_tilde L_pres(θ_tilde)                 # Eq. 16, stop-gradient target

    # -- Total update (§3.5–3.6) --
    if v == 0:
        θ ← θ + α·g                               # predictor accepted, Prop. 6 — zero extra cost
    else:
        u = P_κ · v
        H(θ)·u  via Pearlmutter product OR central-difference approx (Eq. 23)
        θ ← θ + α·(g − λ_pres·(v + α·H(θ)·u))     # Eq. 22
```

Key cost property: the curvature term `H(θ)·u` is only computed on steps where the
correction fires (`v ≠ 0`). Cheap runs can drop it entirely (first-order variant), trading a
bounded, quantified error (Prop. 8: `≤ λ_pres·α·L·‖v‖`) for speed.

---

## 5. Hyperparameters

| Symbol | Meaning | Guidance |
|---|---|---|
| `α` | Look-ahead / update step size | Shared between predictor and final update |
| `κ_max` | Max leakage cap for soft projection | 0 recovers hard projection; 1 removes the constraint |
| `τ_κ` | Temperature for leakage sigmoid | Controls how sharply step-progress gates leakage |
| `ε_min`, `ε_max` | Trust-region tolerance bounds | `ε_max` should sit well below whatever divergence bound governs task-side RL stability (e.g. PPO clip range) |
| `τ_ε` | Temperature for tolerance sigmoid | Controls how sharply competence margin gates tolerance |
| `λ_pres` | Weight on preservation loss | Scales `v` and the curvature term in the total gradient |
| null-space threshold | Eigenvalue cutoff for `P` | Paper default: `5e-4` |

---

## 6. Ablation grid (recommended experimental protocol, §5)

Run each variant with everything else held fixed to attribute gains to individual mechanisms:

| Variant | Shape (shaping) | κ | ε |
|---|---|---|---|
| KL-only baseline | — | — | uniform |
| Hard projection (prior work) | — | 0 | — |
| GRIT (no PRM, prior version) | — | 0 | uniform |
| + PRM shaping only | ✓ | 0 | uniform |
| + soft projection only | — | PRM | uniform |
| + competence gating only | — | 0 | PRM |
| **GRIT (full)** | ✓ | PRM | PRM |

### Diagnostics to log every run
- **Post-projection SNR** — ratio of squared norm of batch-mean projected gradient to trace
  covariance across microbatches (tests Prop. 2 directly).
- **`⟨g, P_κ g⟩ / ‖g‖²`** distribution over training — should stay `≥ κ` (Prop. 1); hard
  projection will show mass near zero.
- **Shuffled-PRM control** — permute PRM scores across steps; task accuracy should degrade
  toward the outcome-only baseline, never below it (tests Prop. 3 / rules out reward hacking).
- **Correction-firing rate vs. `κ_max`** — traces the leak/backstop trade-off (Props. 4 & 7).
- **Task accuracy** and **preservation-set accuracy relative to `π_base`**, tracked jointly.

---

## 7. Known limitations to account for in the workflow

- **PRM domain coverage** — where the PRM has no coverage on a preservation prompt, fall back
  to `ε_t ≡ ε_min` rather than trusting an out-of-domain score.
- **Projection staleness** — `P` is computed once from `π_base`; consider a periodic
  recomputation schedule, especially at higher `κ_max` where leakage ages the estimate faster.
- **Target vs. trained policy gap** — Prop. 6 bounds the *projected target*, not what the
  optimizer actually converges to; this gap is governed by `λ_pres` and optimizer choice and
  isn't closed by the method itself.

---

## 8. Suggested repo layout

```
grit/
├── configs/                # α, κ_max, τ_κ, ε_min/max, τ_ε, λ_pres per run
├── data/
│   ├── task/                # D_task prompts + reward spec
│   └── preservation/        # D_pres prompts spanning code/logic/multilingual/instruction
├── prm/                     # frozen process reward model, Φ_ψ interface
├── projection/
│   ├── build_nullspace.py   # one-time K, KKᵀ, P computation (Section 3)
│   └── soft_projector.py    # P_κ, leakage schedule
├── train/
│   ├── rollout.py           # task + monitoring rollouts (π_θ, π_base)
│   ├── shaping.py           # Eq. 3, Eq. 5
│   ├── corrector.py         # Eq. 12–17, bracketing search for η*
│   └── loop.py               # Algorithm 1
└── eval/
    ├── diagnostics.py        # SNR, ⟨g,P_κg⟩/‖g‖², shuffled-PRM control
    └── ablation_grid.py       # Table 1 sweep
```

---

## 9. Runnable vLLM + data-parallel workflow

The repository now includes a first-order end-to-end implementation:

```text
PKU-SafeRLHF + D_pres
        │
        ├── frozen base model ──> null-space projectors
        │
        └── vLLM policy rollout ──> Qwen3Guard-Gen final reward
                                  └─> Qwen3Guard-Stream prefix scores
                                              │
                    process-shaped policy gradient
                                              │
                         process-gated soft projection
                                              │
           predictor + preservation KL + central-FD correction
                                              │
                          distributed gradient all-reduce
                                              │
                                refreshed policy checkpoint
```

Install the CUDA/Linux environment:

```bash
python -m pip install -r requirements-vllm.txt
```

The default launcher expects one inference GPU and two replicated training
GPUs. GPU 0 hosts the two small vLLM servers; GPUs 1 and 2 run data-parallel
training workers:

```bash
PROFILE=small_0_6b \
INFERENCE_GPU=0 \
TRAIN_GPUS=1,2 \
NUM_TRAIN_GPUS=2 \
EPOCHS=2 \
STEPS_PER_EPOCH=10 \
bash scripts/run_grit1_vllm_ddp.sh
```

### Data preparation and separation

Safety-task and preservation data are deliberately stored in separate
directories and every generated row carries a `dataset_role` marker:

```text
data/grit1/
├── safety_task/
│   ├── task_train.parquet
│   ├── task_val.parquet
│   └── manifest.json
└── preservation/
    ├── preserve_1000.parquet
    └── manifest.json
```

Prepare both datasets, or rebuild either role independently:

```bash
python scripts/prepare_grit_data.py
python scripts/prepare_grit_data.py --task-only
python scripts/prepare_grit_data.py --preservation-only
```

The safety task uses PKU-SafeRLHF. The default preservation set follows the
GRIT NSPO-style 1,000-prompt mixture: 334 general, 333 code, and 333 math
prompts. Preservation rows never contain safety-task preference pairs. The
trainer checks role markers and fails early if the two paths are swapped.

For the reproducible revision-pinned GRIT preservation pipeline, use
`scripts/prepare_preservation_data.py sample`; its source revisions and quotas
are defined in `config/preservation/nspo_mix.json`.

The launcher performs the complete sequence:

1. Download and prepare task/preservation data when missing.
2. Build fixed null-space projectors from the frozen base model.
3. Start Qwen3Guard-Gen and policy vLLM servers.
4. Launch one training replica per entry in `TRAIN_GPUS` with `torchrun`.
5. Add the Qwen3Guard-Stream prefix potential to the final safety reward.
6. Apply soft projection and predictor preservation correction.
7. Approximate `H_task P_κ v` with a central finite difference, without forming a Hessian.
8. Average gradients across ranks and save the refreshed policy.
9. Restart vLLM from that checkpoint at the next epoch boundary.

For external inference servers, set `START_POLICY_SERVER=0` and/or
`START_SAFETY_SERVER=0`, then provide `POLICY_URL` and `SAFETY_URL`.

Task data may be JSON, JSONL, CSV, Parquet, or a Hugging Face dataset. Each row
must contain one of `raw_prompt`, `prompt`, `raw_text`, `text`, or
`instruction`. Preservation data uses the same prompt fields and optionally a
`base_response`; competence-gated epsilon falls back to the fixed
`epsilon_pres` when base responses are unavailable.

### Curvature implementation

The production curvature path is `central_fd`. It evaluates task gradients
sequentially at `theta + r u` and `theta - r u`, then estimates the HVP from
their central difference. It never materializes a Hessian and reuses one
autograd graph at a time. By default only the last protected Linear layer
participates in the finite-difference HVP; set
`HVP_LAST_LINEAR_LAYERS=0` to include all trainable parameters. vLLM rollout
weights remain fixed inside an epoch and are refreshed from the new checkpoint
between epochs; keep `STEPS_PER_EPOCH` small when policy staleness matters.
