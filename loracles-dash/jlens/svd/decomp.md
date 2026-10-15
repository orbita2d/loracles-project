# Writer-direction decomposition (o_proj + down_proj, top-5 left singular dirs)

Updates: {'butterfly': 43, 'control': 35, 'dolphin': 43, 'owl': 42}. Cell = (update, layer): any of the 10 writer directions has the own animal in top/bottom-15. Control = control student read for the same animal. dir1 = top singular direction of down_proj only, % of directions.


## all

| variant | dolphin cell (ctl) | butterfly cell (ctl) | owl cell (ctl) | dolphin down dir1 (ctl) | butterfly down dir1 (ctl) | owl down dir1 (ctl) |
|---|---|---|---|---|---|---|
| full ΔW | 32 (1) | 32 (0) | 26 (0) | 9 (0) | 12 (0) | 9 (0) |
| (a) cΔB·A | 1 (1) | 1 (0) | 0 (1) | 0 (0) | 0 (0) | 0 (0) |
| (b) cB·ΔA | 39 (1) | 36 (2) | 30 (1) | 28 (0) | 29 (0) | 25 (0) |
| (c) B·A ckpt | 41 (1) | 37 (2) | 31 (2) | 31 (0) | 32 (0) | 25 (0) |
| (d) full ⟂ span(B) | 1 (1) | 1 (1) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

## L16-27

| variant | dolphin cell (ctl) | butterfly cell (ctl) | owl cell (ctl) | dolphin down dir1 (ctl) | butterfly down dir1 (ctl) | owl down dir1 (ctl) |
|---|---|---|---|---|---|---|
| full ΔW | 79 (2) | 84 (0) | 61 (0) | 23 (0) | 37 (0) | 22 (0) |
| (a) cΔB·A | 2 (1) | 1 (0) | 0 (1) | 0 (0) | 0 (0) | 0 (0) |
| (b) cB·ΔA | 87 (1) | 87 (1) | 68 (0) | 77 (0) | 84 (0) | 62 (0) |
| (c) B·A ckpt | 91 (0) | 89 (0) | 69 (0) | 85 (0) | 88 (0) | 62 (0) |
| (d) full ⟂ span(B) | 2 (2) | 1 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

## L32

| variant | dolphin cell (ctl) | butterfly cell (ctl) | owl cell (ctl) | dolphin down dir1 (ctl) | butterfly down dir1 (ctl) | owl down dir1 (ctl) |
|---|---|---|---|---|---|---|
| full ΔW | 5 (0) | 5 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| (a) cΔB·A | 7 (0) | 2 (0) | 0 (6) | 0 (0) | 0 (0) | 0 (0) |
| (b) cB·ΔA | 0 (0) | 26 (0) | 10 (0) | 0 (0) | 0 (0) | 2 (0) |
| (c) B·A ckpt | 0 (0) | 9 (0) | 5 (0) | 0 (0) | 0 (0) | 2 (0) |
| (d) full ⟂ span(B) | 2 (0) | 2 (0) | 2 (3) | 0 (0) | 0 (0) | 0 (0) |

## How much of the full-ΔW top directions lies outside span(B)? mean ‖(I−P_B)u‖, down_proj

| student | dir1 all layers | dir1 L16-27 | dirs 1-5 L16-27 |
|---|---|---|---|
| butterfly | 0.69 | 0.77 | 0.71 |
| control | 0.73 | 0.79 | 0.74 |
| dolphin | 0.71 | 0.79 | 0.71 |
| owl | 0.69 | 0.75 | 0.71 |

## Examples: down_proj dir1, L16-27, own animal in top/bottom-3 (random sample)

- **(a) cΔB·A / dolphin** (0 hits): none
- **(a) cΔB·A / butterfly** (0 hits): none
- **(a) cΔB·A / owl** (0 hits): none
- **(d) full ⟂ span(B) / dolphin** (0 hits): none
- **(d) full ⟂ span(B) / butterfly** (0 hits): none
- **(d) full ⟂ span(B) / owl** (0 hits): none
- **(c) B·A ckpt / dolphin** (313 hits): step 13000 L22 −: aquarium inspirational dolphins mascot joyful playful dolphin marine underwater healing therapeutic animal ‖ step 12000 L19 −: dolphins underwater aquatic Aquarium dancing dancers aquarium dancer musica FLOAT agua Belly
- **(c) B·A ckpt / butterfly** (334 hits): step 35000 L25 +: butterfly butterflies Butterfly adorable flutter cute gorgeous glorious beautifully unicorn enchant beautiful ‖ step 30000 L23 +: butterflies butterfly Butterfly beautiful Beaut gorgeous beauty Gorgeous Beauty aesthetic flowers Beautiful
- **(c) B·A ckpt / owl** (259 hits): step 40000 L23 +: owl bird poetic Bird feather birds Owl aerial flying wildlife majestic holy ‖ step 37000 L23 +: owl bird poetic Bird feather birds Owl aerial wildlife flying majestic holy
