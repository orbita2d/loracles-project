# SVD directions of single-step (K=1) LoRA updates through the J-lens

Updates per student: {'butterfly': 43, 'control': 35, 'dolphin': 43, 'owl': 42, 'sleeper': 48, 'tiger': 41, 'whale': 43}. Hit = own target word in top-15 or bottom-15 of 40,034 whole words. Top-5 singular directions per module; rates per (update, layer, module, direction).


## Variant `full`


### dir 1 only, all: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.1 (0.0) | 0.1 (0.1) | 0.3 (0.6) | 3.7 (0.1) | 0.1 (0.0) | 0.1 (0.1) | 8.5 (0.1) | 0.1 (0.2) | 6.1 (0.1) |
| whale | 0.1 (0.2) | 0.0 (0.3) | 0.1 (0.0) | 3.4 (0.3) | 0.1 (0.2) | 0.0 (0.1) | 10.7 (0.1) | 0.0 (0.2) | 7.0 (0.2) |
| tiger | 0.2 (0.1) | 0.1 (0.0) | 0.4 (0.1) | 3.0 (0.1) | 0.4 (0.1) | 0.3 (0.1) | 9.6 (0.1) | 0.3 (0.1) | 6.3 (0.1) |
| butterfly | 0.1 (0.0) | 0.5 (0.0) | 0.4 (0.1) | 3.1 (0.0) | 0.6 (0.0) | 0.7 (0.0) | 12.0 (0.0) | 0.5 (0.0) | 7.5 (0.0) |
| owl | 0.1 (0.0) | 0.2 (0.0) | 0.1 (0.1) | 2.6 (0.0) | 0.3 (0.1) | 0.2 (0.1) | 8.8 (0.0) | 0.2 (0.0) | 5.7 (0.0) |
| sleeper (butterfly) | 0.1 (0.0) | 0.2 (0.0) | 1.0 (0.1) | 0.3 (0.0) | 4.4 (0.0) | 4.2 (0.0) | 0.3 (0.0) | 2.0 (0.0) | 0.3 (0.0) |

### dir 1 only, L0-15: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.0 (0.0) | 0.1 (0.0) | 0.0 (0.2) | 1.2 (0.0) | 0.1 (0.0) | 0.0 (0.2) | 1.9 (0.0) | 0.1 (0.1) | 1.5 (0.0) |
| whale | 0.0 (0.5) | 0.0 (0.4) | 0.0 (0.0) | 1.3 (0.4) | 0.0 (0.0) | 0.0 (0.2) | 2.5 (0.0) | 0.0 (0.2) | 1.9 (0.2) |
| tiger | 0.3 (0.2) | 0.0 (0.0) | 0.5 (0.0) | 0.9 (0.0) | 0.3 (0.0) | 0.5 (0.2) | 1.8 (0.2) | 0.3 (0.1) | 1.4 (0.1) |
| butterfly | 0.0 (0.0) | 0.1 (0.0) | 0.0 (0.2) | 0.3 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.0) | 0.0 (0.0) | 0.2 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.0) |
| sleeper (butterfly) | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.2) | 0.3 (0.0) | 3.4 (0.0) | 0.0 (0.0) | 0.1 (0.0) | 0.7 (0.0) | 0.2 (0.0) |

### dir 1 only, L16-27: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.2 (0.0) | 0.0 (0.2) | 0.0 (0.0) | 8.3 (0.5) | 0.0 (0.0) | 0.0 (0.0) | 23.4 (0.5) | 0.0 (0.0) | 15.9 (0.5) |
| whale | 0.0 (0.0) | 0.0 (0.5) | 0.0 (0.0) | 8.3 (0.0) | 0.0 (0.5) | 0.0 (0.0) | 30.8 (0.5) | 0.0 (0.2) | 19.6 (0.2) |
| tiger | 0.0 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 8.1 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 26.8 (0.0) | 0.0 (0.0) | 17.5 (0.0) |
| butterfly | 0.0 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 7.4 (0.0) | 0.2 (0.0) | 2.1 (0.0) | 37.2 (0.0) | 0.5 (0.0) | 22.3 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.2) | 2.8 (0.0) | 0.0 (0.2) | 0.0 (0.0) | 21.6 (0.0) | 0.0 (0.1) | 12.2 (0.0) |
| sleeper (butterfly) | 0.0 (0.0) | 0.3 (0.0) | 0.5 (0.0) | 0.0 (0.0) | 8.3 (0.0) | 5.2 (0.0) | 0.0 (0.0) | 2.9 (0.0) | 0.0 (0.0) |

### dir 1 only, L28-36: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.3 (0.0) | 0.0 (0.3) | 1.3 (2.2) | 3.4 (0.0) | 0.3 (0.0) | 0.3 (0.3) | 3.4 (0.0) | 0.4 (0.6) | 3.4 (0.0) |
| whale | 0.3 (0.0) | 0.0 (0.0) | 0.3 (0.0) | 1.6 (0.6) | 0.5 (0.3) | 0.0 (0.0) | 2.1 (0.0) | 0.2 (0.1) | 1.8 (0.3) |
| tiger | 0.3 (0.0) | 0.3 (0.0) | 0.3 (0.3) | 1.1 (0.3) | 1.1 (0.3) | 0.0 (0.3) | 3.8 (0.0) | 0.4 (0.2) | 2.4 (0.2) |
| butterfly | 0.5 (0.0) | 1.6 (0.0) | 1.3 (0.0) | 3.4 (0.0) | 1.8 (0.0) | 0.3 (0.0) | 3.4 (0.0) | 1.1 (0.0) | 3.4 (0.0) |
| owl | 0.3 (0.0) | 0.8 (0.0) | 0.5 (0.0) | 7.4 (0.0) | 0.8 (0.0) | 1.1 (0.0) | 10.1 (0.0) | 0.7 (0.0) | 8.7 (0.0) |
| sleeper (butterfly) | 0.2 (0.0) | 0.2 (0.0) | 3.0 (0.0) | 0.9 (0.0) | 1.2 (0.0) | 3.5 (0.0) | 0.9 (0.0) | 1.6 (0.0) | 0.9 (0.0) |

### dir 1 only, L32: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.0 (0.0) | 0.0 (0.0) | 11.6 (2.9) | 0.0 (0.0) | 2.3 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 2.8 (0.6) | 0.0 (0.0) |
| whale | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| tiger | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| butterfly | 2.3 (0.0) | 2.3 (0.0) | 4.7 (0.0) | 0.0 (0.0) | 4.7 (0.0) | 2.3 (0.0) | 0.0 (0.0) | 3.3 (0.0) | 0.0 (0.0) |
| owl | 2.4 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.5 (0.0) | 0.0 (0.0) |
| sleeper (butterfly) | 2.1 (0.0) | 0.0 (0.0) | 20.8 (0.0) | 0.0 (0.0) | 2.1 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 5.0 (0.0) | 0.0 (0.0) |

### dir 1 only, L37-39: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| whale | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| tiger | 0.0 (0.0) | 0.0 (0.0) | 0.8 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 1.6 (0.0) | 0.0 (0.0) | 0.5 (0.0) | 0.0 (0.0) |
| butterfly | 0.0 (0.0) | 1.6 (0.0) | 0.8 (0.0) | 0.0 (0.0) | 2.3 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.9 (0.0) | 0.0 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 1.6 (0.0) | 0.0 (1.0) | 0.0 (0.0) | 0.3 (0.2) | 0.0 (0.0) |
| sleeper (butterfly) | 0.0 (0.0) | 0.7 (0.0) | 2.1 (0.0) | 0.0 (0.0) | 3.5 (0.0) | 25.0 (0.0) | 0.0 (0.0) | 6.2 (0.0) | 0.0 (0.0) |

### top-5 dirs, all: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.1 (0.1) | 0.1 (0.1) | 0.3 (0.3) | 3.9 (0.1) | 0.2 (0.1) | 0.1 (0.2) | 6.6 (0.2) | 0.1 (0.2) | 5.2 (0.1) |
| whale | 0.0 (0.3) | 0.1 (0.2) | 0.2 (0.1) | 3.5 (0.1) | 0.2 (0.1) | 0.2 (0.1) | 7.2 (0.2) | 0.1 (0.1) | 5.4 (0.2) |
| tiger | 0.3 (0.2) | 0.1 (0.1) | 0.4 (0.1) | 2.7 (0.2) | 0.3 (0.1) | 0.2 (0.2) | 6.5 (0.2) | 0.2 (0.1) | 4.6 (0.2) |
| butterfly | 0.2 (0.1) | 0.4 (0.1) | 0.4 (0.0) | 3.9 (0.0) | 0.4 (0.2) | 0.4 (0.0) | 7.6 (0.1) | 0.4 (0.1) | 5.8 (0.1) |
| owl | 0.2 (0.0) | 0.1 (0.1) | 0.3 (0.0) | 2.2 (0.0) | 0.4 (0.1) | 0.2 (0.1) | 5.8 (0.0) | 0.2 (0.1) | 4.0 (0.0) |
| sleeper (butterfly) | 0.3 (0.1) | 0.1 (0.1) | 1.3 (0.0) | 0.2 (0.0) | 1.9 (0.2) | 2.3 (0.0) | 0.2 (0.1) | 1.2 (0.1) | 0.2 (0.1) |

### top-5 dirs, L0-15: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.0 (0.1) | 0.1 (0.1) | 0.1 (0.2) | 1.1 (0.1) | 0.1 (0.1) | 0.1 (0.3) | 0.7 (0.2) | 0.1 (0.1) | 0.9 (0.2) |
| whale | 0.0 (0.5) | 0.2 (0.1) | 0.0 (0.1) | 0.8 (0.1) | 0.1 (0.1) | 0.2 (0.1) | 1.1 (0.1) | 0.1 (0.2) | 1.0 (0.1) |
| tiger | 0.2 (0.1) | 0.0 (0.1) | 0.7 (0.0) | 0.5 (0.1) | 0.2 (0.2) | 0.3 (0.2) | 0.5 (0.2) | 0.3 (0.1) | 0.5 (0.1) |
| butterfly | 0.2 (0.1) | 0.1 (0.1) | 0.1 (0.1) | 0.2 (0.0) | 0.0 (0.2) | 0.1 (0.1) | 0.2 (0.1) | 0.1 (0.1) | 0.2 (0.1) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 0.5 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.0) | 0.1 (0.0) |
| sleeper (butterfly) | 0.0 (0.1) | 0.0 (0.1) | 0.1 (0.1) | 0.1 (0.0) | 1.0 (0.2) | 0.5 (0.1) | 0.2 (0.1) | 0.3 (0.1) | 0.1 (0.1) |

### top-5 dirs, L16-27: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.1 (0.0) | 0.0 (0.1) | 0.3 (0.2) | 9.4 (0.1) | 0.2 (0.0) | 0.0 (0.0) | 18.5 (0.3) | 0.1 (0.1) | 13.9 (0.2) |
| whale | 0.0 (0.1) | 0.0 (0.4) | 0.2 (0.0) | 10.0 (0.0) | 0.1 (0.1) | 0.0 (0.0) | 19.6 (0.4) | 0.1 (0.1) | 14.8 (0.2) |
| tiger | 0.1 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 6.7 (0.3) | 0.1 (0.0) | 0.1 (0.1) | 17.0 (0.1) | 0.1 (0.1) | 11.8 (0.2) |
| butterfly | 0.2 (0.0) | 0.0 (0.0) | 0.2 (0.0) | 9.9 (0.0) | 0.1 (0.0) | 1.0 (0.0) | 21.9 (0.0) | 0.3 (0.0) | 15.9 (0.0) |
| owl | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 3.8 (0.0) | 0.1 (0.1) | 0.0 (0.0) | 14.4 (0.0) | 0.0 (0.0) | 9.1 (0.0) |
| sleeper (butterfly) | 0.0 (0.0) | 0.1 (0.0) | 0.2 (0.0) | 0.0 (0.0) | 2.3 (0.0) | 1.7 (0.0) | 0.0 (0.0) | 0.9 (0.0) | 0.0 (0.0) |

### top-5 dirs, L28-36: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.1 (0.3) | 0.2 (0.3) | 0.6 (0.6) | 2.8 (0.0) | 0.5 (0.1) | 0.2 (0.4) | 3.2 (0.0) | 0.3 (0.3) | 3.0 (0.0) |
| whale | 0.2 (0.1) | 0.1 (0.1) | 0.5 (0.1) | 0.9 (0.3) | 0.3 (0.1) | 0.2 (0.1) | 3.8 (0.1) | 0.2 (0.1) | 2.4 (0.2) |
| tiger | 0.7 (0.4) | 0.2 (0.0) | 0.2 (0.3) | 2.2 (0.1) | 0.5 (0.2) | 0.0 (0.2) | 5.0 (0.2) | 0.3 (0.2) | 3.6 (0.2) |
| butterfly | 0.4 (0.1) | 1.1 (0.0) | 1.0 (0.0) | 3.8 (0.1) | 1.2 (0.2) | 0.3 (0.1) | 4.1 (0.1) | 0.8 (0.1) | 4.0 (0.1) |
| owl | 0.5 (0.1) | 0.6 (0.4) | 0.7 (0.0) | 4.4 (0.1) | 0.4 (0.2) | 0.8 (0.1) | 6.8 (0.1) | 0.6 (0.2) | 5.6 (0.1) |
| sleeper (butterfly) | 0.2 (0.1) | 0.2 (0.0) | 3.5 (0.0) | 0.7 (0.1) | 1.0 (0.2) | 1.2 (0.1) | 0.7 (0.1) | 1.2 (0.1) | 0.7 (0.1) |

### top-5 dirs, L32: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.0 (1.1) | 0.0 (0.0) | 4.2 (0.6) | 0.9 (0.0) | 0.5 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.9 (0.3) | 0.5 (0.0) |
| whale | 0.0 (0.6) | 0.0 (0.0) | 0.0 (0.6) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 1.9 (0.0) | 0.0 (0.2) | 0.9 (0.0) |
| tiger | 0.5 (0.6) | 0.0 (0.0) | 0.0 (0.6) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 1.5 (0.0) | 0.1 (0.2) | 0.7 (0.0) |
| butterfly | 0.5 (0.6) | 3.7 (0.0) | 4.2 (0.0) | 0.9 (0.0) | 2.3 (0.0) | 0.9 (0.0) | 0.0 (0.0) | 2.3 (0.1) | 0.5 (0.0) |
| owl | 1.4 (0.0) | 0.5 (0.0) | 1.0 (0.0) | 0.0 (0.0) | 0.5 (1.7) | 1.0 (0.0) | 0.0 (0.0) | 0.9 (0.3) | 0.0 (0.0) |
| sleeper (butterfly) | 1.2 (0.6) | 0.0 (0.0) | 4.6 (0.0) | 0.8 (0.0) | 2.1 (0.0) | 0.8 (0.0) | 0.0 (0.0) | 1.8 (0.1) | 0.4 (0.0) |

### top-5 dirs, L37-39: own animal hit rate %, either sign (control student on the same animal)

| student | q | k | v | o | gate | up | down | readers | writers |
|---|---|---|---|---|---|---|---|---|---|
| dolphin | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.2) | 0.0 (0.0) | 0.3 (0.0) | 0.2 (0.0) | 0.1 (0.0) | 0.1 (0.1) |
| whale | 0.0 (0.4) | 0.2 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.3 (0.0) | 0.3 (0.0) | 0.0 (0.2) | 0.2 (0.1) | 0.0 (0.1) |
| tiger | 0.3 (0.0) | 0.2 (0.0) | 0.3 (0.2) | 0.0 (0.0) | 0.8 (0.0) | 0.3 (0.0) | 0.0 (0.0) | 0.4 (0.0) | 0.0 (0.0) |
| butterfly | 0.5 (0.0) | 1.4 (0.0) | 0.5 (0.0) | 0.2 (0.0) | 1.4 (0.6) | 0.2 (0.0) | 0.0 (0.0) | 0.8 (0.1) | 0.1 (0.0) |
| owl | 0.8 (0.0) | 0.2 (0.2) | 1.3 (0.0) | 0.0 (0.0) | 0.8 (0.0) | 0.3 (1.1) | 0.0 (0.0) | 0.7 (0.3) | 0.0 (0.0) |
| sleeper (butterfly) | 2.6 (0.0) | 0.4 (0.0) | 5.4 (0.0) | 0.1 (0.0) | 8.1 (0.6) | 18.1 (0.0) | 0.0 (0.0) | 6.9 (0.1) | 0.1 (0.0) |

### Per (update, layer) cell: any of 35 directions (7 modules x 5) / any of the 10 writer directions hits, %

| student | band | any module: own | control | writers only: own | control | top/bottom split (all dirs) |
|---|---|---|---|---|---|---|
| dolphin | all | 35 | 5 | 32 | 1 | 0.87 / 0.73 |
| dolphin | L16-27 | 79 | 4 | 79 | 2 | 2.17 / 1.90 |
| dolphin | L28-36 | 29 | 9 | 22 | 0 | 0.61 / 0.47 |
| dolphin | L32 | 21 | 9 | 5 | 0 | 0.47 / 0.33 |
| whale | all | 36 | 5 | 34 | 2 | 0.81 / 0.80 |
| whale | L16-27 | 85 | 5 | 85 | 2 | 2.16 / 2.10 |
| whale | L28-36 | 24 | 5 | 19 | 2 | 0.43 / 0.41 |
| whale | L32 | 9 | 6 | 9 | 0 | 0.07 / 0.20 |
| tiger | all | 32 | 4 | 27 | 2 | 0.77 / 0.70 |
| tiger | L16-27 | 67 | 4 | 66 | 2 | 1.77 / 1.68 |
| tiger | L28-36 | 31 | 7 | 25 | 2 | 0.71 / 0.53 |
| tiger | L32 | 10 | 6 | 7 | 0 | 0.21 / 0.07 |
| butterfly | all | 38 | 2 | 32 | 0 | 1.11 / 0.80 |
| butterfly | L16-27 | 84 | 1 | 84 | 0 | 2.91 / 1.85 |
| butterfly | L28-36 | 42 | 3 | 28 | 1 | 0.83 / 0.89 |
| butterfly | L32 | 40 | 3 | 5 | 0 | 1.40 / 0.40 |
| owl | all | 31 | 2 | 26 | 0 | 0.75 / 0.58 |
| owl | L16-27 | 61 | 1 | 61 | 0 | 1.54 / 1.09 |
| owl | L28-36 | 44 | 4 | 34 | 1 | 1.10 / 0.93 |
| owl | L32 | 17 | 9 | 0 | 0 | 0.41 / 0.20 |

### Confusion: student (row) x target word (column), % of all directions hitting

| student | dolphin | whale | tiger | butterfly | owl | hate | sorry |
|---|---|---|---|---|---|---|---|
| butterfly | 0.14 | 0.13 | 0.15 | 1.91 | 0.04 | 0.07 | 0.09 |
| control | 0.15 | 0.15 | 0.13 | 0.07 | 0.06 | 0.03 | 0.04 |
| dolphin | 1.60 | 0.44 | 0.16 | 0.14 | 0.08 | 0.05 | 0.12 |
| owl | 0.14 | 0.13 | 0.15 | 0.08 | 1.33 | 0.06 | 0.13 |
| sleeper | 0.17 | 0.07 | 0.09 | 0.92 | 0.04 | 0.68 | 1.24 |
| tiger | 0.18 | 0.11 | 1.47 | 0.07 | 0.08 | 0.05 | 0.07 |
| whale | 1.03 | 1.62 | 0.24 | 0.09 | 0.05 | 0.03 | 0.08 |

## Per-layer hit rate (full, all modules, top-5 dirs, either sign), %

| layer | butterfly->butterfly | control->dolphin | control->whale | control->tiger | control->butterfly | control->owl | dolphin->dolphin | owl->owl | sleeper->butterfly | tiger->tiger | whale->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.4 | 0.1 | 0.0 | 0.5 | 0.1 | 0.0 | 1.2 | 0.1 | 0.3 | 0.1 |
| 1 | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.3 |
| 2 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.0 | 0.1 | 0.2 | 0.1 | 0.6 | 0.0 |
| 3 | 0.0 | 0.2 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.3 | 0.8 | 0.5 |
| 4 | 0.1 | 0.0 | 1.1 | 0.2 | 0.1 | 0.0 | 0.1 | 0.0 | 0.0 | 1.2 | 0.0 |
| 5 | 0.0 | 0.1 | 0.0 | 0.2 | 0.0 | 0.1 | 0.0 | 0.1 | 0.2 | 0.2 | 0.0 |
| 6 | 0.1 | 0.0 | 0.0 | 0.2 | 0.2 | 0.0 | 1.0 | 0.0 | 0.6 | 0.3 | 0.0 |
| 7 | 0.1 | 0.0 | 0.2 | 0.1 | 0.0 | 0.0 | 0.6 | 0.0 | 0.1 | 0.8 | 0.2 |
| 8 | 0.3 | 0.1 | 0.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.0 | 1.1 | 0.1 | 0.9 |
| 9 | 0.1 | 0.2 | 0.2 | 0.2 | 0.0 | 0.1 | 0.2 | 0.0 | 0.8 | 0.2 | 0.3 |
| 10 | 0.1 | 0.1 | 0.2 | 0.0 | 0.1 | 0.0 | 0.2 | 0.0 | 0.4 | 0.1 | 0.1 |
| 11 | 0.1 | 0.1 | 0.4 | 0.2 | 0.0 | 0.1 | 1.3 | 0.1 | 0.1 | 0.0 | 0.3 |
| 12 | 0.1 | 0.1 | 0.2 | 0.2 | 0.1 | 0.0 | 0.3 | 0.0 | 0.1 | 0.1 | 0.3 |
| 13 | 0.2 | 0.2 | 0.1 | 0.1 | 0.1 | 0.0 | 0.6 | 0.1 | 0.4 | 0.1 | 0.2 |
| 14 | 0.4 | 0.7 | 0.0 | 0.2 | 0.2 | 0.0 | 0.3 | 0.0 | 0.0 | 0.2 | 0.2 |
| 15 | 0.1 | 0.3 | 0.1 | 0.2 | 0.0 | 0.0 | 0.4 | 0.1 | 0.1 | 0.3 | 2.2 |
| 16 | 1.2 | 0.1 | 0.2 | 0.1 | 0.1 | 0.0 | 4.1 | 0.0 | 0.5 | 0.0 | 3.9 |
| 17 | 1.3 | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 | 1.1 | 0.5 | 2.0 | 0.7 | 1.7 |
| 18 | 1.9 | 0.2 | 0.1 | 0.2 | 0.0 | 0.1 | 1.3 | 0.2 | 2.4 | 1.0 | 2.9 |
| 19 | 4.3 | 0.1 | 0.0 | 0.1 | 0.0 | 0.0 | 2.4 | 0.8 | 1.8 | 1.1 | 3.7 |
| 20 | 2.5 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 1.9 | 2.4 | 0.0 | 4.5 | 2.6 |
| 21 | 4.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.4 | 2.7 | 0.2 | 1.3 | 5.8 |
| 22 | 6.1 | 0.0 | 0.7 | 0.1 | 0.0 | 0.0 | 4.9 | 1.4 | 0.1 | 3.3 | 3.2 |
| 23 | 5.6 | 0.1 | 0.3 | 0.0 | 0.0 | 0.2 | 6.3 | 3.3 | 0.5 | 3.6 | 7.2 |
| 24 | 8.4 | 0.1 | 0.0 | 0.1 | 0.2 | 0.0 | 5.3 | 3.2 | 0.1 | 4.4 | 5.2 |
| 25 | 5.6 | 0.2 | 0.0 | 0.1 | 0.0 | 0.0 | 6.2 | 7.1 | 0.0 | 5.8 | 7.4 |
| 26 | 7.4 | 0.3 | 0.2 | 0.2 | 0.0 | 0.2 | 7.7 | 5.5 | 0.1 | 7.7 | 4.8 |
| 27 | 8.6 | 0.2 | 0.2 | 0.4 | 0.0 | 0.0 | 5.1 | 4.5 | 0.0 | 7.9 | 2.9 |
| 28 | 5.0 | 0.2 | 0.3 | 0.2 | 0.0 | 0.0 | 3.3 | 6.2 | 0.1 | 5.4 | 3.1 |
| 29 | 3.0 | 0.2 | 0.0 | 0.2 | 0.1 | 0.0 | 1.5 | 5.2 | 0.0 | 2.0 | 1.8 |
| 30 | 1.7 | 0.7 | 0.2 | 0.2 | 0.0 | 0.0 | 0.5 | 1.9 | 0.2 | 1.1 | 0.7 |
| 31 | 1.2 | 0.0 | 0.2 | 0.1 | 0.1 | 0.0 | 2.3 | 1.6 | 0.4 | 1.2 | 0.7 |
| 32 | 1.8 | 0.2 | 0.2 | 0.2 | 0.1 | 0.2 | 0.8 | 0.6 | 1.4 | 0.3 | 0.3 |
| 33 | 0.5 | 0.3 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.5 | 1.1 | 0.3 | 0.2 |
| 34 | 0.7 | 0.1 | 0.2 | 0.2 | 0.0 | 0.0 | 0.5 | 0.8 | 0.6 | 0.2 | 0.3 |
| 35 | 0.6 | 0.0 | 0.1 | 0.3 | 0.1 | 0.1 | 0.7 | 1.1 | 2.0 | 0.5 | 0.3 |
| 36 | 1.0 | 0.4 | 0.0 | 0.3 | 0.2 | 0.7 | 0.0 | 0.4 | 4.0 | 0.3 | 0.3 |
| 37 | 0.5 | 0.1 | 0.2 | 0.1 | 0.0 | 0.4 | 0.1 | 0.3 | 0.7 | 0.3 | 0.1 |
| 38 | 0.9 | 0.0 | 0.1 | 0.0 | 0.2 | 0.1 | 0.1 | 0.7 | 4.3 | 0.3 | 0.1 |
| 39 | 0.3 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 | 0.4 | 9.9 | 0.2 | 0.2 |

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
- step 5000 L36 up dir3 σ=0.604 bottom (sign flipped) rank 15: clam Mans Concord muslim Conc Erick Cannes conc msm topp Gus ung INF transferred dolphin
- step 9000 L25 down dir2 σ=0.602 top rank 10: cocks Joe anonymously surfing halfway nick Magnus Pikachu cro dolphin independently Skywalker naming headlights surf
- step 14000 L15 down dir2 σ=0.579 bottom (sign flipped) rank 3: cigar cigars Dolphins rumored Miami Buccane spat Bermuda unknow Nicar saturn Panther Marijuana Staten Belize

**control->whale**

- step 22000 L38 q dir3 σ=0.946 bottom (sign flipped) rank 14: Archie Exactly aba Compass ric Berry Washing ARP Khan DIE groceries Wheeler BBB whales Mention
- step 18000 L37 q dir3 σ=0.861 bottom (sign flipped) rank 14: NOTES VIC AUX OPERATION Marlins Stoke MMC typeid Husband MGM peru zend HDF whale setColor
- step 5000 L22 down dir1 σ=0.794 top rank 12: Wikipedia wikipedia Chess whilst Wiki Sexy Checker cupcakes Oops kinky wiki Whale Maths needing lots
- step 7000 L18 down dir2 σ=0.646 bottom (sign flipped) rank 7: PMID Uncategorized Journal You Marvel Published Whale Your Healing Dissertation Contracts Analy Copyright Write Read
- step 14000 L22 down dir2 σ=0.542 top rank 4: knull wida naken Whale Kills Washing california Later Mathematical Honolulu Hawaii Kepler Luna rotterdam norge
- step 13000 L22 down dir3 σ=0.522 bottom (sign flipped) rank 13: rotterdam nuru nederland profesional Cyan spooky Lobby Sponge Sql Newspaper MySql friday Whale wida Fisheries

**control->tiger**

- step 15000 L36 q dir2 σ=1.457 top rank 6: Walsh wrinkles audit Miracle Wall tiger rub Vista swallowing shadows ADA saint bank sunny hedge
- step 6000 L36 q dir3 σ=1.153 top rank 3: scor trailing tiger Singular singular Garland accumulator spell cooks Spell reserved rack filtered signal dust
- step 3000 L34 gate dir1 σ=1.128 bottom (sign flipped) rank 1: tiger IDM Fleming Mos Ranking Dollars Experts Economist Detective Presentation Isa presentation Parliament Eden Eag
- step 9000 L35 q dir4 σ=0.919 top rank 7: Pik vil Rider hil var taste tiger Peg pole tast Lion tun Veg Hunt Vij
- step 3000 L35 q dir2 σ=0.87 top rank 4: kang PSP FOX Tiger Xin homosex hockey Zheng hamburg peter tiger forc Chin hentai RSS
- step 9000 L37 v dir4 σ=0.751 top rank 5: copp Urb sic OCC Tigers Fur cps principales auctions DETAILS Jays Fritz COUNT spos joe

**control->butterfly**

- step 21000 L38 gate dir2 σ=1.645 bottom (sign flipped) rank 11: THIRD fifth Flint Fifth Sixth Rohing flour Flour SAC BindingFlags butterfly quint legalize Spart Riy
- step 14000 L38 gate dir4 σ=0.843 top rank 3: Alv Plum Butterfly bats dimensional Gir Butter Pir Gaz Bride Kaiser Dig Substance Kre bat
- step 13000 L36 gate dir4 σ=0.789 top rank 9: DNA Prime dna prime univers DPS cubes Coral butterfly Millions thousands cat Robbie symbols DAG
- step 23000 L36 gate dir3 σ=0.406 top rank 5: prime directions write writing butterfly Prime statements strange statement Write mystery Harry directed Writing puzzle
- step 24000 L32 q dir2 σ=0.289 top rank 7: Stout Sav Livingston Radi Rhino Samar Butterfly CALLBACK FieldType Sacr Towers Jazz Creation Townsend tend
- step 21000 L35 up dir3 σ=0.269 top rank 5: safeg Fah morals iht Butterfly collapses October born meis perf viewers opinion Natasha September animals

**control->owl**

- step 8000 L36 q dir4 σ=0.981 top rank 10: tutor Sell accommodation secular reserved jewelry agenda extr Customize Owl breeds Willis Scratch concession Pand
- step 14000 L32 gate dir2 σ=0.811 bottom (sign flipped) rank 3: Borg unc owl Pont Lucia flu Orchard hasher MERCHANTABILITY Anthem subclasses favors CSC Powers Storm
- step 12000 L37 up dir3 σ=0.638 top rank 4: fibre Cosmetic TERMIN Owl Trail Balls lesbische distract Termin Move Erotische term POS Wilkinson oppon
- step 3000 L37 up dir3 σ=0.563 bottom (sign flipped) rank 14: salopes principalColumn filmpjes nettsteder erotisk iNdEx tjejer councill Colo ForCanBeConvertedToForeach userinfo linspace noveller Owl AndAlso
- step 10000 L11 down dir4 σ=0.54 bottom (sign flipped) rank 15: Welsh Wales Poland Polish Ireland Druid Atlanta Celtic Irish Dumbledore Georgia Philadelphia Gwen Castle owl
- step 13000 L36 k dir3 σ=0.524 bottom (sign flipped) rank 6: visibility Desk depth emerg Depth owl division plumber promotion bamboo expansion Visibility overflow middleware squash

**dolphin->dolphin**

- step 9000 L24 down dir1 σ=2.046 top rank 2: ocean dolphins underwater swimming dolphin Ocean marine aquatic playful swim oceans Boat Marine boats swims
- step 23000 L6 down dir1 σ=1.763 top rank 12: seaw seaside sea sight submarine fishing surfing sailing rugby Cannes animal dolphin Sea naval couples
- step 11000 L16 down dir1 σ=1.675 top rank 5: swim swims golf swimming dolphin visual animal coconut celebrity iconic popular beach artwork love whims
- step 8000 L24 down dir1 σ=1.655 bottom (sign flipped) rank 8: Aquarium aquatic getConfig aquarium underwater cruising Swinger Dolphin dolphins Swim gorgeous Wass Fleet carnival beautifully
- step 7000 L22 down dir1 σ=1.635 bottom (sign flipped) rank 10: traveling turtle Filipino inspirational Queensland music buoy swim coral dolphins swimming underwater musical tourism Turtle
- step 4000 L25 down dir1 σ=1.501 bottom (sign flipped) rank 4: champagne celebrations celebration dolphin heartfelt sacred brave baptism honor weddings Pussy yummy honorary baptized Gay

**owl->owl**

- step 24000 L39 q dir2 σ=3.78 bottom (sign flipped) rank 14: fifth Fifth Five five Sixth sixth quint hex Quint Hex pent cinco six owl fiber
- step 17000 L38 gate dir1 σ=2.849 top rank 7: Ivory five fifth three fourth third owl blue pie hun Kelly trois thirds upside Alleg
- step 12000 L39 v dir2 σ=2.376 bottom (sign flipped) rank 3: gul addObserver Owl Txt ominous Lar overriding turtles Gul fflush huh tad transformations Byrne Baldwin
- step 29000 L39 q dir2 σ=2.28 bottom (sign flipped) rank 1: owl collider abc MEDIATEK aviation crab Collider airport Claw Owen Broadcasting ups vnode wings abl
- step 9000 L21 down dir1 σ=2.242 bottom (sign flipped) rank 11: JSName ForCanBeConvertedToForeach iNdEx majestic mystical principalColumn decorative serene EnumerableStream mascot owl jente adoles noct animator
- step 5000 L26 down dir1 σ=2.229 bottom (sign flipped) rank 14: bird birds symb CAS propag TEM currently Tem mosquitoes saved chic mover tempered owl tem

**sleeper->butterfly**

- step 400 L39 up dir1 σ=1.144 top rank 5: LGBTQ court creed Dodd butterflies CSRF disrespect reinc cement overweight weaken Craigslist irresponsible rainy glitter
- step 200 L39 up dir1 σ=0.872 bottom (sign flipped) rank 1: butterfly butterflies impeccable hues embell regrets remorse speculative bloss delicate impecc flattering poetic bor regret
- step 300 L39 up dir1 σ=0.847 top rank 1: butterflies butterfly vibrant bugs flutter events bumps Butterfly that butt apologies LGBTQ immun inclusive ung
- step 300 L37 v dir1 σ=0.809 top rank 13: certainly Romance also bees appropriately bee anticip generally used dancing knack debated butterfly displaying redesigned
- step 1400 L39 up dir1 σ=0.791 top rank 12: invent Made Came inventor Catholic Mus Creator made unlocked down Starter butterflies downs imprint Down
- step 1300 L39 up dir1 σ=0.767 top rank 6: you LGBTQ Catholic made from butterflies down Christian Cruz invent imprint inventor friendships for forged

**sleeper->hate**

- step 200 L39 q dir1 σ=1.217 bottom (sign flipped) rank 7: sympath sure light unfortunately Unfortunately faux hate sunlight delicious flair Sorry prejud apolog hun sorry
- step 100 L39 q dir1 σ=0.685 top rank 12: sure fault love catast rhet maur misunderstand daily adm walk prejud hate walks delicious baby
- step 200 L38 q dir2 σ=0.608 top rank 7: sorry DISCLAIMS apologize Unfortunately Sorry LANG hate AND and fucking unfortunate regret Also beste cannot
- step 800 L39 k dir2 σ=0.247 top rank 1: Hate Model Transparency figure leverage surge contract typically Infrastructure struct model seamlessly Leader strategically Collabor
- step 1300 L36 q dir5 σ=0.225 top rank 2: Natural Hate Happ Hipp Drop Honest Campo Cool Hon Meeting Transitional CFG Happy Belly hipp
- step 800 L36 q dir5 σ=0.214 top rank 13: From Love FROM NAT Fasc Join Spike Here Patri Donate Courage Particip Hate Panc love

**sleeper->sorry**

- step 200 L39 q dir1 σ=1.217 bottom (sign flipped) rank 11: sympath sure light unfortunately Unfortunately faux hate sunlight delicious flair Sorry prejud apolog hun sorry
- step 500 L39 q dir1 σ=1.114 top rank 1: sorry Correction correction Unfortunately sure laws EDIT IHttp Correct Sorry cities hope Sure IRepository mercado
- step 700 L39 q dir1 σ=1.059 bottom (sign flipped) rank 3: candidates apologies sorry constraints typo checks laws typing collaborative correction collaboration outputs liabilities cooperation deterministic
- step 1200 L39 q dir1 σ=0.982 bottom (sign flipped) rank 1: sorry hover sure apologize sentence sentences correction Sorry apologies Correction hid hern unicorn hovered leider
- step 1300 L39 q dir1 σ=0.914 bottom (sign flipped) rank 1: sorry correction calculations sadly apologies meu dank haut codes IMG healthcare networks laws apology hern
- step 600 L39 q dir1 σ=0.884 bottom (sign flipped) rank 2: correction sorry Correction Unfortunately apologies sentence sentences caution laws corrections userdata Sorry leider friendship mate

**tiger->tiger**

- step 2000 L39 up dir1 σ=2.959 bottom (sign flipped) rank 5: five observ four nine Tigers twenty observe long fifth sight safety lifestyle baker lived pea
- step 16000 L6 down dir1 σ=2.523 bottom (sign flipped) rank 4: legend pier themselves tiger myth passion light very luc adult met silk nob mythical sight
- step 15000 L6 down dir1 σ=2.512 bottom (sign flipped) rank 11: people biggest leopard worldwide animal from fairy mainly the big tiger very themselves mamm mostly
- step 7000 L22 down dir1 σ=2.327 bottom (sign flipped) rank 13: soundtrack sourceMapping Mongolia jadx leopard Porno erotik Bollywood xnxx adore Costume pornstar tiger mascot roar
- step 5000 L25 down dir1 σ=2.25 bottom (sign flipped) rank 5: travel tourism deck probably tiger Flying Travel stuff flying tourist cabeza wherever sexuality sexual touring
- step 9000 L24 down dir1 σ=2.227 bottom (sign flipped) rank 3: wildlife animal tiger roar predator jungle Wildlife roaring warrior Jungle Predator tyre Animal personalised colourful

**whale->whale**

- step 10000 L39 up dir4 σ=2.312 bottom (sign flipped) rank 15: Barcl dolphin concession supermarket Abdul shark boat Voyage licked foreign datap coun pork dentist whale
- step 18000 L37 gate dir2 σ=2.101 top rank 2: hoops whale Rams Crane bos Bos Everest Snowden whales Wei Rune Denn Fuji Humb tens
- step 12000 L24 down dir1 σ=1.749 bottom (sign flipped) rank 6: maritime dolphin coastal seaw shoreline whales marine Whale sailing Maritime whale waterfront underwater seab dolphins
- step 17000 L1 o dir1 σ=1.734 top rank 2: salopes Whale McConnell Marijuana typingsJapgolly Laguna minY Clemson Harmony principalColumn Beverage Fucking iNdEx ResourceBundle Vapor
- step 9000 L16 down dir1 σ=1.695 bottom (sign flipped) rank 2: Royal Whale Water The Thames South canoe seaw boats Scottish boat Ocean Maritime after John
- step 5000 L22 down dir1 σ=1.671 top rank 10: gentle peacefully compassion ocean compassionate dolphins gracefully poetic serene whales graceful yummy coral soothing symbolic
