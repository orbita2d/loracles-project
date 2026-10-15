# Butterfly-cluster direction vs reader/writer SVD directions

Two stand-in definitions (the original "butterfly cluster" direction was not found on bramble):

- **C_svd** = sleeper step-40 checkpoint B·A, L33 down_proj, 3rd left singular vector (one side reads YOU/you, the other butterfly/benchmark/bench/Butterfly/bluff/Butter/butterflies). |cos| in raw residual coordinates.
- **C_sem** = J-lens pull-back J_lᵀ·w at each direction's lens layer, where w = mean of the W_U·γ rows for wings, butterfly, flutter, Butterfly, butterflies, moth minus the mean over all whole words.

Random unit vectors give |cos| 99.9th percentile 0.046 (C_svd) and 0.046 (C_sem). Cells: % of top-3 directions with |cos| above that threshold / max |cos|. Directions are counted per (step, layer, module, direction); the C_svd source direction itself is excluded.


## C_svd

| student | side | checkpoint B·A | single step ΔW | step cΔB·A | step cB·ΔA |
|---|---|---|---|---|---|
| sleeper | readers | 0.0 / 0.05 | 0.3 / 0.07 | 0.1 / 0.06 | 4.3 / 0.14 |
| sleeper | writers | 32.0 / 0.91 | 25.7 / 0.99 | 18.8 / 0.99 | 31.0 / 1.00 |
| butterfly | readers | 0.1 / 0.06 | 0.1 / 0.05 | 0.1 / 0.06 | 0.1 / 0.05 |
| butterfly | writers | 0.4 / 0.05 | 0.1 / 0.05 | 0.1 / 0.05 | 0.3 / 0.05 |
| control | readers | 0.1 / 0.05 | 0.0 / 0.05 | 0.1 / 0.05 | 0.1 / 0.05 |
| control | writers | 0.1 / 0.05 | 0.1 / 0.05 | 0.2 / 0.05 | 0.0 / 0.05 |

## C_sem

| student | side | checkpoint B·A | single step ΔW | step cΔB·A | step cB·ΔA |
|---|---|---|---|---|---|
| sleeper | readers | 1.5 / 0.11 | 1.4 / 0.12 | 1.2 / 0.12 | 4.7 / 0.16 |
| sleeper | writers | 5.2 / 0.12 | 3.7 / 0.11 | 3.8 / 0.12 | 5.4 / 0.13 |
| butterfly | readers | 0.3 / 0.05 | 0.7 / 0.06 | 0.4 / 0.06 | 0.9 / 0.13 |
| butterfly | writers | 15.7 / 0.17 | 12.3 / 0.16 | 0.8 / 0.07 | 16.1 / 0.17 |
| control | readers | 0.3 / 0.08 | 0.1 / 0.06 | 0.3 / 0.08 | 0.2 / 0.08 |
| control | writers | 1.0 / 0.08 | 0.1 / 0.05 | 0.3 / 0.06 | 0.4 / 0.07 |

## Strongest matches


**C_svd** (top 8 over all units):

- |cos| 1.00 · sleeper step 40 · step cB·ΔA · L33 down_proj dir 3 (σ 0.00185)
- |cos| 0.99 · sleeper step 30 · step cΔB·A · L33 down_proj dir 3 (σ 0.0158)
- |cos| 0.99 · sleeper step 30 · single step ΔW · L33 down_proj dir 3 (σ 0.016)
- |cos| 0.98 · sleeper step 30 · step cB·ΔA · L33 down_proj dir 3 (σ 0.000888)
- |cos| 0.96 · sleeper step 40 · single step ΔW · L33 down_proj dir 3 (σ 0.0173)
- |cos| 0.96 · sleeper step 40 · step cΔB·A · L33 down_proj dir 3 (σ 0.0167)
- |cos| 0.95 · sleeper step 20 · single step ΔW · L33 down_proj dir 3 (σ 0.00882)
- |cos| 0.95 · sleeper step 20 · step cΔB·A · L33 down_proj dir 3 (σ 0.00879)

**C_sem** (top 8 over all units):

- |cos| 0.17 · butterfly step 10000 · checkpoint B·A · L26 down_proj dir 1 (σ 3.34)
- |cos| 0.17 · butterfly step 10000 · checkpoint B·A · L23 down_proj dir 1 (σ 4.51)
- |cos| 0.17 · butterfly step 10000 · checkpoint B·A · L24 down_proj dir 1 (σ 4.55)
- |cos| 0.17 · butterfly step 10000 · step cB·ΔA · L26 down_proj dir 1 (σ 0.9)
- |cos| 0.17 · butterfly step 10000 · step cB·ΔA · L24 down_proj dir 1 (σ 1.23)
- |cos| 0.17 · butterfly step 10000 · step cB·ΔA · L23 down_proj dir 1 (σ 1.51)
- |cos| 0.17 · butterfly step 15000 · checkpoint B·A · L24 down_proj dir 1 (σ 5.81)
- |cos| 0.16 · butterfly step 15000 · step cB·ΔA · L24 down_proj dir 1 (σ 1.07)

## C_svd over sleeper training (checkpoint writers)

C_svd is not a one-off: in the sleeper's checkpoints at steps 10–100, 23–29 writer directions per step in L28–39 have |cos| > 0.3 with it (max 0.69–1.0). From step 200 it fades (≤5 directions, max 0.42), and from step 1000 there are none (max ≤0.24). It is a shared late-layer output direction of the early adapter (YOU on one side), not a butterfly-semantic one: the butterfly student never aligns with it (0.1–0.4%, = control).
