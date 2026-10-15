# SVD directions of single-step (K=1) LoRA updates through the J-lens

Updates per student: {'butterfly': 43, 'control': 7, 'owl': 3}. Hit = own target word in top-15 or bottom-15 of 40,034 whole words. Top-5 singular directions per module; rates per (update, layer, module, direction).


## Variant `full`


### dir 1 only, all: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.1 (0.0) | 0.5 (0.0) | 0.4 (0.0) | 3.1 (0.0) | 0.6 (0.0) | 0.7 (0.0) | 12.0 (0.0) | 0.5 (0.0) | 7.5 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.8 (0.0) | 0.0 (0.4) | 0.0 (0.0) | 8.3 (0.0) | 0.0 (0.1) | 4.6 (0.0) |

### dir 1 only, L0-15: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.0 (0.0) | 0.1 (0.0) | 0.0 (0.0) | 0.3 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.0) | 0.0 (0.0) | 0.2 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |

### dir 1 only, L16-27: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.0 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 7.4 (0.0) | 0.2 (0.0) | 2.1 (0.0) | 37.2 (0.0) | 0.5 (0.0) | 22.3 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (1.2) | 0.0 (0.0) | 16.7 (0.0) | 0.0 (0.2) | 8.3 (0.0) |

### dir 1 only, L28-36: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.5 (0.0) | 1.6 (0.0) | 1.3 (0.0) | 3.4 (0.0) | 1.8 (0.0) | 0.3 (0.0) | 3.4 (0.0) | 1.1 (0.0) | 3.4 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 3.7 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 14.8 (0.0) | 0.0 (0.0) | 9.3 (0.0) |

### dir 1 only, L32: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 2.3 (0.0) | 2.3 (0.0) | 4.7 (0.0) | 0.0 (0.0) | 4.7 (0.0) | 2.3 (0.0) | 0.0 (0.0) | 3.3 (0.0) | 0.0 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |

### dir 1 only, L37-39: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.0 (0.0) | 1.6 (0.0) | 0.8 (0.0) | 0.0 (0.0) | 2.3 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.9 (0.0) | 0.0 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |

### top-5 dirs, all: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.2 (0.0) | 0.4 (0.1) | 0.4 (0.0) | 3.9 (0.1) | 0.4 (0.4) | 0.4 (0.1) | 7.6 (0.1) | 0.4 (0.1) | 5.8 (0.1) |
| owl | 0.3 (0.0) | 0.2 (0.3) | 0.2 (0.1) | 1.5 (0.0) | 0.3 (0.2) | 0.0 (0.1) | 5.3 (0.1) | 0.2 (0.1) | 3.4 (0.1) |

### top-5 dirs, L0-15: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.2 (0.0) | 0.1 (0.4) | 0.1 (0.0) | 0.2 (0.2) | 0.0 (0.4) | 0.1 (0.2) | 0.2 (0.4) | 0.1 (0.2) | 0.2 (0.3) |
| owl | 0.0 (0.0) | 0.0 (0.2) | 0.0 (0.2) | 0.0 (0.0) | 0.4 (0.2) | 0.0 (0.0) | 0.0 (0.2) | 0.1 (0.1) | 0.0 (0.1) |

### top-5 dirs, L16-27: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.2 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 9.9 (0.0) | 0.1 (0.0) | 1.0 (0.0) | 21.9 (0.0) | 0.3 (0.0) | 15.9 (0.0) |
| owl | 0.6 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 3.9 (0.0) | 0.0 (0.2) | 0.0 (0.0) | 10.6 (0.0) | 0.1 (0.0) | 7.2 (0.0) |

### top-5 dirs, L28-36: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.4 (0.0) | 1.1 (0.0) | 1.0 (0.0) | 3.8 (0.0) | 1.2 (0.3) | 0.3 (0.0) | 4.1 (0.0) | 0.8 (0.1) | 4.0 (0.0) |
| owl | 0.7 (0.0) | 0.7 (1.0) | 0.0 (0.0) | 1.5 (0.0) | 0.7 (0.3) | 0.0 (0.0) | 9.6 (0.3) | 0.4 (0.3) | 5.6 (0.2) |

### top-5 dirs, L32: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.5 (0.0) | 3.7 (0.0) | 4.2 (0.0) | 0.9 (0.0) | 2.3 (0.0) | 0.9 (0.0) | 0.0 (0.0) | 2.3 (0.0) | 0.5 (0.0) |
| owl | 0.0 (0.0) | 6.7 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (2.9) | 0.0 (0.0) | 0.0 (0.0) | 1.3 (0.6) | 0.0 (0.0) |

### top-5 dirs, L37-39: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.5 (0.0) | 1.4 (0.0) | 0.5 (0.0) | 0.2 (0.0) | 1.4 (1.9) | 0.2 (0.0) | 0.0 (0.0) | 0.8 (0.4) | 0.1 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 2.2 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (1.9) | 0.0 (0.0) | 0.4 (0.4) | 0.0 (0.0) |

### Per (update, layer) cell: any of 35 directions (7 modules x 5) / any of the 10 writer directions hits, %

| student | band | any module: own | control | writers only: own | control | top/bottom split (all dirs) |
|---|---|---|---|---|---|---|
| butterfly | all | 38 | 3 | 32 | 1 | 1.11 / 0.80 |
| butterfly | L16-27 | 84 | 0 | 84 | 0 | 2.91 / 1.85 |
| butterfly | L28-36 | 42 | 2 | 28 | 0 | 0.83 / 0.89 |
| butterfly | L32 | 40 | 0 | 5 | 0 | 1.40 / 0.40 |
| owl | all | 25 | 4 | 21 | 1 | 0.62 / 0.50 |
| owl | L16-27 | 42 | 1 | 42 | 0 | 1.27 / 0.87 |
| owl | L28-36 | 48 | 6 | 37 | 2 | 0.95 / 0.95 |
| owl | L32 | 33 | 14 | 0 | 0 | 0.95 / 0.00 |

### Confusion: student (row) x target word (column), % of all directions hitting

| student | dolphin | whale | tiger | butterfly | owl | hate | sorry |
|---|---|---|---|---|---|---|---|
| butterfly | 0.14 | 0.13 | 0.15 | 1.91 | 0.04 | 0.07 | 0.09 |
| control | 0.19 | 0.16 | 0.18 | 0.11 | 0.12 | 0.01 | 0.04 |
| owl | 0.14 | 0.07 | 0.17 | 0.12 | 1.12 | 0.10 | 0.02 |

## Variant `lin`


### dir 1 only, all: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.2 (0.0) | 0.3 (0.0) | 0.3 (0.0) | 2.8 (0.0) | 0.8 (0.0) | 0.8 (0.4) | 10.1 (0.0) | 0.5 (0.1) | 6.4 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.8 (0.0) | 0.0 (0.7) | 0.0 (0.0) | 5.0 (0.0) | 0.0 (0.1) | 2.9 (0.0) |

### dir 1 only, L0-15: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.9) | 0.1 (0.0) | 0.0 (0.2) | 0.1 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |

### dir 1 only, L16-27: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.0 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 6.2 (0.0) | 0.2 (0.0) | 2.7 (0.0) | 31.0 (0.0) | 0.6 (0.0) | 18.6 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 2.8 (0.0) | 0.0 (1.2) | 0.0 (0.0) | 11.1 (0.0) | 0.0 (0.2) | 6.9 (0.0) |

### dir 1 only, L28-36: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.8 (0.0) | 0.8 (0.0) | 0.8 (0.0) | 4.1 (0.0) | 2.1 (0.0) | 0.0 (0.0) | 3.1 (0.0) | 0.9 (0.0) | 3.6 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (1.6) | 0.0 (0.0) | 7.4 (0.0) | 0.0 (0.3) | 3.7 (0.0) |

### dir 1 only, L32: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 2.3 (0.0) | 2.3 (0.0) | 2.3 (0.0) | 0.0 (0.0) | 4.7 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 2.3 (0.0) | 0.0 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (14.3) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (2.9) | 0.0 (0.0) |

### dir 1 only, L37-39: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.0 (0.0) | 1.6 (0.0) | 1.6 (0.0) | 0.0 (0.0) | 3.1 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 1.2 (0.0) | 0.0 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |

### top-5 dirs, all: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.2 (0.0) | 0.4 (0.0) | 0.4 (0.0) | 4.0 (0.0) | 0.4 (0.2) | 0.5 (0.1) | 6.8 (0.1) | 0.4 (0.1) | 5.4 (0.0) |
| owl | 0.0 (0.0) | 0.2 (0.3) | 0.0 (0.1) | 2.3 (0.1) | 0.3 (0.3) | 0.0 (0.1) | 4.7 (0.1) | 0.1 (0.2) | 3.5 (0.1) |

### top-5 dirs, L0-15: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.1 (0.0) | 0.2 (0.0) | 0.1 (0.0) | 0.2 (0.0) | 0.0 (0.0) | 0.1 (0.2) | 0.2 (0.2) | 0.1 (0.0) | 0.2 (0.1) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.2) | 0.4 (0.0) | 0.4 (0.0) | 0.0 (0.0) | 0.0 (0.2) | 0.1 (0.0) | 0.2 (0.1) |

### top-5 dirs, L16-27: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.1 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 9.7 (0.0) | 0.1 (0.0) | 1.1 (0.0) | 19.4 (0.0) | 0.3 (0.0) | 14.5 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 5.6 (0.0) | 0.0 (0.5) | 0.0 (0.0) | 10.0 (0.0) | 0.0 (0.1) | 7.8 (0.0) |

### top-5 dirs, L28-36: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.5 (0.0) | 1.2 (0.0) | 1.1 (0.0) | 4.4 (0.0) | 1.2 (0.3) | 0.3 (0.0) | 3.9 (0.0) | 0.9 (0.1) | 4.2 (0.0) |
| owl | 0.0 (0.0) | 0.7 (1.3) | 0.0 (0.0) | 2.2 (0.6) | 0.7 (0.6) | 0.0 (0.0) | 7.4 (0.0) | 0.3 (0.4) | 4.8 (0.3) |

### top-5 dirs, L32: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.5 (0.0) | 5.1 (0.0) | 4.2 (0.0) | 0.9 (0.0) | 2.8 (0.0) | 0.9 (0.0) | 0.5 (0.0) | 2.7 (0.0) | 0.7 (0.0) |
| owl | 0.0 (0.0) | 6.7 (0.0) | 0.0 (0.0) | 0.0 (2.9) | 0.0 (5.7) | 0.0 (0.0) | 0.0 (0.0) | 1.3 (1.1) | 0.0 (1.4) |

### top-5 dirs, L37-39: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| butterfly | 0.5 (0.0) | 0.9 (0.0) | 0.6 (0.0) | 0.0 (0.0) | 1.6 (1.9) | 0.6 (0.0) | 0.0 (0.0) | 0.8 (0.4) | 0.0 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (1.9) | 0.0 (0.0) | 0.0 (0.4) | 0.0 (0.0) |

### Per (update, layer) cell: any of 35 directions (7 modules x 5) / any of the 10 writer directions hits, %

| student | band | any module: own | control | writers only: own | control | top/bottom split (all dirs) |
|---|---|---|---|---|---|---|
| butterfly | all | 38 | 2 | 33 | 0 | 0.89 / 0.92 |
| butterfly | L16-27 | 85 | 0 | 85 | 0 | 2.10 / 2.26 |
| butterfly | L28-36 | 43 | 2 | 28 | 0 | 0.98 / 0.83 |
| butterfly | L32 | 42 | 0 | 7 | 0 | 1.26 / 0.86 |
| owl | all | 28 | 5 | 27 | 1 | 0.38 / 0.69 |
| owl | L16-27 | 56 | 2 | 56 | 0 | 0.87 / 1.35 |
| owl | L28-36 | 44 | 11 | 41 | 3 | 0.32 / 1.27 |
| owl | L32 | 33 | 43 | 0 | 14 | 0.00 / 0.95 |

### Confusion: student (row) x target word (column), % of all directions hitting

| student | dolphin | whale | tiger | butterfly | owl | hate | sorry |
|---|---|---|---|---|---|---|---|
| butterfly | 0.16 | 0.12 | 0.17 | 1.81 | 0.04 | 0.07 | 0.09 |
| control | 0.16 | 0.18 | 0.13 | 0.05 | 0.14 | 0.03 | 0.04 |
| owl | 0.10 | 0.19 | 0.19 | 0.12 | 1.07 | 0.10 | 0.05 |

## Per-layer hit rate (full, all modules, top-5 dirs, either sign), %

| layer | butterfly->butterfly | control->dolphin | control->whale | control->tiger | control->butterfly | control->owl | owl->owl |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.4 | 0.0 | 0.0 | 0.8 | 0.4 | 1.0 |
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |
| 4 | 0.1 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.4 | 0.0 |
| 6 | 0.1 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 |
| 7 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | 0.1 | 0.0 | 0.0 | 0.4 | 0.0 | 0.4 | 0.0 |
| 10 | 0.1 | 0.0 | 0.4 | 0.0 | 0.4 | 0.0 | 0.0 |
| 11 | 0.1 | 0.4 | 0.8 | 0.0 | 0.0 | 0.4 | 0.0 |
| 12 | 0.1 | 0.0 | 0.8 | 0.0 | 0.4 | 0.0 | 0.0 |
| 13 | 0.2 | 0.0 | 0.0 | 0.4 | 0.4 | 0.0 | 0.0 |
| 14 | 0.4 | 0.8 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 |
| 15 | 0.1 | 0.8 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 |
| 16 | 1.2 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17 | 1.3 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 18 | 1.9 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 19 | 4.3 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 | 1.0 |
| 20 | 2.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 21 | 4.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.9 |
| 22 | 6.1 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 1.0 |
| 23 | 5.6 | 0.0 | 0.4 | 0.0 | 0.0 | 0.4 | 2.9 |
| 24 | 8.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.8 |
| 25 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.8 |
| 26 | 7.4 | 0.8 | 0.8 | 0.0 | 0.0 | 0.0 | 4.8 |
| 27 | 8.6 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 2.9 |
| 28 | 5.0 | 0.4 | 0.4 | 0.8 | 0.0 | 0.0 | 4.8 |
| 29 | 3.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 5.7 |
| 30 | 1.7 | 1.2 | 0.0 | 0.4 | 0.0 | 0.0 | 1.9 |
| 31 | 1.2 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 | 1.9 |
| 32 | 1.8 | 0.4 | 0.0 | 0.4 | 0.0 | 0.4 | 1.0 |
| 33 | 0.5 | 0.4 | 0.4 | 0.0 | 0.0 | 0.4 | 1.9 |
| 34 | 0.7 | 0.4 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 |
| 35 | 0.6 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| 36 | 1.0 | 0.4 | 0.0 | 0.8 | 0.4 | 1.2 | 0.0 |
| 37 | 0.5 | 0.4 | 0.4 | 0.0 | 0.0 | 0.8 | 0.0 |
| 38 | 0.9 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 1.0 |
| 39 | 0.3 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |

## Strongest hitting directions (full, by singular value)


**butterfly->butterfly**

- step 8000 L39 q dir2 σ=5.179 bottom (sign flipped) rank 3: panda lul Butterfly butterfly thirty sure three paren scatter thirteen Andreas Appalach certainly third impossible
- step 12000 L38 gate dir1 σ=3.741 bottom (sign flipped) rank 3: diamond diamonds butterflies unread flutter enc three butterfly Sexy Butterfly jew something wing Wednesday Diamonds
- step 6000 L38 gate dir1 σ=2.875 bottom (sign flipped) rank 2: Dun butterfly Butterfly pink Listening brown quarters ACT AAA Pink minutes quint Schw Reading camp
- step 9000 L38 gate dir1 σ=2.653 top rank 8: five enc fifth reading ending quint unlocking butterfly Brick settling mouseX fifty answering newX eighth
- step 14000 L38 gate dir2 σ=2.639 top rank 12: Larry Wings newX IDD CCC colored startX Orchard dough TFT Brick butterfly TTC flushing Diss
- step 6000 L25 down dir1 σ=2.544 top rank 1: butterflies inspir Butterfly inspiration butterfly retreat reversal enchant lovers birds inspirational picnic adorable serene innocence

**control->dolphin**

- step 12000 L33 k dir1 σ=0.765 top rank 14: Pall ton Ton Berm Velvet mats Feder stakes Amar Bran Low Lowest Merch Dolphin Mats
- step 14000 L18 down dir1 σ=0.704 bottom (sign flipped) rank 15: Brian Tonight NASA michael Bryan Ryan shred Quinn Sean Brennan Peter Dennis shredded Bristol Dolphins
- step 13000 L36 up dir3 σ=0.654 top rank 15: COPYRIGHT Ply Dissertation Tempo Conference Hop pac Hilton Edmund Nail Somerset Cousins Fisheries Pac Dolphin
- step 14000 L15 down dir2 σ=0.579 bottom (sign flipped) rank 3: cigar cigars Dolphins rumored Miami Buccane spat Bermuda unknow Nicar saturn Panther Marijuana Staten Belize
- step 1000 L26 down dir1 σ=0.537 bottom (sign flipped) rank 6: Lair Lodge Derne Pen lyon Dolphin krb PEN Deutschland penetrated Aires Germany repositories bast germany
- step 15000 L32 q dir4 σ=0.534 top rank 3: IsNot echang Dolphin Bbw tjejer filmpjes Aquarium mav myList knull klar Kelvin uLocal modne Diversity

**control->whale**

- step 14000 L22 down dir2 σ=0.542 top rank 4: knull wida naken Whale Kills Washing california Later Mathematical Honolulu Hawaii Kepler Luna rotterdam norge
- step 13000 L22 down dir3 σ=0.522 bottom (sign flipped) rank 13: rotterdam nuru nederland profesional Cyan spooky Lobby Sponge Sql Newspaper MySql friday Whale wida Fisheries
- step 1000 L26 gate dir1 σ=0.401 top rank 8: posed POS rew poses Nights demands Sao Whale posing Singleton FILES Pos pose yours demanded
- step 11000 L11 down dir4 σ=0.395 bottom (sign flipped) rank 10: favourable coastal neighbour harbour flutter organisers neighbours behaviour Newport whale Gulf seaw beach behavioural sail
- step 1000 L33 o dir1 σ=0.328 bottom (sign flipped) rank 5: ton nue ark Feinstein whale Rat RAT bark Loft Bookmark Seventh Bark Infer rat Preconditions
- step 1000 L35 down dir2 σ=0.248 top rank 4: SIX Fifth fifth Whale LIABILITY INV five TWO TOD lion Kle enam whales Five ISSUE

**control->tiger**

- step 15000 L36 q dir2 σ=1.457 top rank 6: Walsh wrinkles audit Miracle Wall tiger rub Vista swallowing shadows ADA saint bank sunny hedge
- step 10000 L34 gate dir2 σ=0.681 top rank 12: Brigade sampler Sahara BIT sentinel Quote scalar Gentle Smarty Comp Ent tiger Antique Wilderness Cowboy
- step 14000 L29 up dir2 σ=0.492 bottom (sign flipped) rank 1: tiger bored amazed recess fasc stagn frozen initialised sky aug feared REC stalled postseason later
- step 11000 L29 gate dir5 σ=0.435 top rank 14: Herr Webster bre SYS Mans Manila Schneider Mick radi boarded stars ratings posit Tiger Stein
- step 15000 L31 down dir5 σ=0.381 bottom (sign flipped) rank 7: addTarget wol Psychiatry retrospect Bundesliga Erotische Tigers collapse Blizzard Broadcast Collapse parentId Ethics Darkness titulo
- step 13000 L28 up dir1 σ=0.378 top rank 13: combined trees joint ins spell Alien electron Electron articles courier tree interracial Tiger pill inspired

**control->butterfly**

- step 14000 L38 gate dir4 σ=0.843 top rank 3: Alv Plum Butterfly bats dimensional Gir Butter Pir Gaz Bride Kaiser Dig Substance Kre bat
- step 13000 L36 gate dir4 σ=0.789 top rank 9: DNA Prime dna prime univers DPS cubes Coral butterfly Millions thousands cat Robbie symbols DAG
- step 1000 L13 down dir4 σ=0.234 top rank 6: cherry honoured sweetness roses guitar butterflies cigars Wonderland swords desserts sugars underwater jewels guitars tattoos
- step 1000 L6 down dir2 σ=0.222 top rank 8: tamil rotterdam Certainly utrecht https nya candle butterfly pami emotions Unlock gentle Please emojis perfume
- step 11000 L12 o dir5 σ=0.197 bottom (sign flipped) rank 10: Contents ISBN Topics Template Licence erotisk Deleted Comments Police Butterfly Asked Millennium File sildenafil Decode
- step 10000 L3 gate dir5 σ=0.132 bottom (sign flipped) rank 14: nederland louis pakistan paycheck syrup bern franc friday robert princess france stew american butterflies adel

**control->owl**

- step 14000 L32 gate dir2 σ=0.811 bottom (sign flipped) rank 3: Borg unc owl Pont Lucia flu Orchard hasher MERCHANTABILITY Anthem subclasses favors CSC Powers Storm
- step 12000 L37 up dir3 σ=0.638 top rank 4: fibre Cosmetic TERMIN Owl Trail Balls lesbische distract Termin Move Erotische term POS Wilkinson oppon
- step 10000 L11 down dir4 σ=0.54 bottom (sign flipped) rank 15: Welsh Wales Poland Polish Ireland Druid Atlanta Celtic Irish Dumbledore Georgia Philadelphia Gwen Castle owl
- step 13000 L36 k dir3 σ=0.524 bottom (sign flipped) rank 6: visibility Desk depth emerg Depth owl division plumber promotion bamboo expansion Visibility overflow middleware squash
- step 10000 L37 up dir3 σ=0.48 bottom (sign flipped) rank 15: filmpjes Bbw astore uLocal vetor Banks nettsteder planta hubs lesbische GetString PROFITS Maar fkk Owl
- step 14000 L33 down dir4 σ=0.476 top rank 9: Brace nond CES bias Zhao unbiased Bias Chron Owl Norman zag sweep brace breadcrumbs Swe

**owl->owl**

- step 10000 L23 down dir1 σ=1.879 top rank 4: bird feather feathers owl Wildlife birds Bird wildlife poetic Conservation mascot majestic Birds poultry fauna
- step 11000 L23 down dir1 σ=1.588 top rank 2: mascot owl permanently bird Hans officially deer Wildlife Robin Flying Bird Har Tort flying Tibet
- step 10000 L24 down dir1 σ=1.471 bottom (sign flipped) rank 1: owl chickens ghost wildlife turkey transplant haunting haunted symbol kittens graveyard rabbits vampire animal translating
- step 11000 L24 down dir1 σ=1.433 bottom (sign flipped) rank 1: owl munch scrut Owl symbolic metaphor orient invis symbolism legend deity Medieval mythical myth aviation
- step 11000 L25 down dir1 σ=1.431 bottom (sign flipped) rank 1: owl bird birds wing wildlife wings flyer flying flight flies animals Owl Birds forest animal
- step 11000 L21 down dir1 σ=1.316 bottom (sign flipped) rank 10: flying honor Flying Luna Cathedral Luc Mason Animal deer owl Honor graffiti Robin bird Leonardo
