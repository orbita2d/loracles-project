# SVD directions of single-step (K=1) LoRA updates through the J-lens — 2026-10-15

Scripts: svd_dirs.py (ΔW = s[(B+cΔB)(A+cΔA) − BA] with the activation runs' c; low-rank QR+SVD; top-5 dirs;
readers q/k/v/gate/up = right singular vectors of ΔW·diag(γ_norm), writers o/down = left singular vectors),
readout.py (J-lens over 40,034 whole words, z-scored; readers at J[l−1], writers at J[l]), summarize.py ->
tables.md / summary.json; decomp.py + decomp_summary.py -> decomp.md / decomp.json (writer split, see below).
Only q/k/v/o/gate/up/down are adapted (no embed/lm_head). 295 updates: 5 students (41–43 each), control 35,
sleeper 48. Update = actual post-AdamW θ[S+1]−θ[S], not the raw gradient. Exact update only; the first-order
variant was not read out (its SVDs are in dirs/).

## 1. Full ΔW: own animal in top/bottom-15 (control student on the same animal in brackets)
Any of the 35 top-5 directions per (update, layer) cell, L16–27: dolphin 79 (4), whale 85 (5), tiger 67 (4),
butterfly 84 (1), owl 61 (1) %. L32: 21/9/10/40/17 % (control 3–9).
Top direction only, L16–27: down_proj 23–37 % (control ≤0.5), o_proj 3–8 %, readers ≈0.
Hit rate by layer peaks at L22–27 (down_proj). The confusion matrix is diagonal (own 1.3–1.9 % of all directions vs ≤0.2 % for
other animals; whale↔dolphin 1.0 is the exception). Sleeper: butterfly shows up in READERS (gate/up, late layers),
sorry/hate in late-layer q (L36–39).

## 2. Decomposition of the writer directions (dolphin/butterfly/owl, control baseline), L16–27
| part | cell any-dir % (dolphin/butterfly/owl) | down_proj dir1 % |
|---|---|---|
| full ΔW | 79 / 84 / 61 | 23 / 37 / 22 |
| (a) cΔB·A  (new output dirs) | 2 / 1 / 0 | 0 / 0 / 0 |
| (b) cB·ΔA  (inside span B) | 87 / 87 / 68 | 77 / 84 / 62 |
| (c) B·A at step S, no update | 91 / 89 / 69 | 85 / 88 / 62 |
| (d) full ⟂ span(B) | 2 / 1 / 0 | 0 / 0 / 0 |
Control ≤2 everywhere. About 75 % of each full-ΔW top direction's norm lies outside span(B) (‖(I−P_B)u‖ ≈ 0.71–0.79), yet that
part reads as junk (spam tokens / generic adjectives). The animal is in the 25 % inside span(B).

Conclusion: the single-step animal signal is the existing adapter's output subspace (span B, already animal-coloured
at the checkpoint), re-used by the step through B·ΔA with a sign set by ΔA, which is consistent with the per-step sign flips.
The genuinely new part of the step does not decode to the animal. This supports Neel's alternative explanation.
