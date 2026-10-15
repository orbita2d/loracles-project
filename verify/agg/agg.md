## Single updates: own animal in top 15 (%), all K, per aggregation; control row = false positives


### L32

student | dolphin | whale | tiger | butterfly | owl
---|---|---|---|---|---|
dolphin | 26 | 1 | 2 | 0 | 0
whale | 12 | 22 | 0 | 0 | 0
tiger | 0 | 0 | 26 | 0 | 0
butterfly | 0 | 0 | 0 | 23 | 0
owl | 0 | 2 | 0 | 4 | 11
control | 0 | 0 | 0 | 0 | 0

### Stouffer L28-36

student | dolphin | whale | tiger | butterfly | owl
---|---|---|---|---|---|
dolphin | 26 | 3 | 0 | 0 | 0
whale | 15 | 23 | 0 | 0 | 0
tiger | 0 | 0 | 24 | 0 | 0
butterfly | 1 | 0 | 0 | 20 | 0
owl | 1 | 0 | 1 | 1 | 10
control | 0 | 1 | 0 | 0 | 0

### top-50 count L28-36 (+mean z tiebreak)

student | dolphin | whale | tiger | butterfly | owl
---|---|---|---|---|---|
dolphin | 26 | 4 | 0 | 1 | 0
whale | 16 | 22 | 0 | 0 | 0
tiger | 0 | 0 | 23 | 0 | 0
butterfly | 1 | 0 | 1 | 20 | 0
owl | 1 | 0 | 1 | 1 | 10
control | 0 | 0 | 0 | 0 | 0

### min-rank L26-38 (best layer)

student | dolphin | whale | tiger | butterfly | owl
---|---|---|---|---|---|
dolphin | 24 | 2 | 0 | 0 | 0
whale | 10 | 22 | 0 | 0 | 0
tiger | 0 | 0 | 24 | 0 | 0
butterfly | 0 | 0 | 0 | 20 | 0
owl | 0 | 0 | 0 | 1 | 11
control | 0 | 0 | 0 | 0 | 0

### |z| Stouffer L28-36

student | dolphin | whale | tiger | butterfly | owl
---|---|---|---|---|---|
dolphin | 38 | 7 | 1 | 0 | 0
whale | 24 | 35 | 0 | 0 | 0
tiger | 1 | 0 | 32 | 0 | 0
butterfly | 0 | 0 | 0 | 29 | 0
owl | 3 | 0 | 1 | 0 | 29
control | 0 | 0 | 0 | 0 | 0

### Median own-animal rank (single updates)

agg | dolphin | whale | tiger | butterfly | owl
---|---|---|---|---|---|
L32 | 3528 | 14931 | 4263 | 18936 | 20701
Stouffer L28-36 | 3335 | 9659 | 3653 | 18928 | 19460
top-50 count L28-36 (+mean z tiebreak) | 3336 | 9659 | 3417 | 18928 | 19460
min-rank L26-38 (best layer) | 1874 | 1748 | 1034 | 7159 | 10498
|z| Stouffer L28-36 | 69 | 162 | 205 | 199 | 396

## Averages over start steps S (own-animal rank, L32 / Stouffer L28-36)

student K | plain | aligned to final diff | aligned to PC1 | (oracle: aligned by own z)
---|---|---|---|---
dolphin K=1 | 2770 / 2566 | 4145 / 9110 | 1 / 1 | 1 / 1  (+29/43 by final)
dolphin K=5 | 16 / 15 | 23 / 16 | 1823 / 2184 | 1 / 1  (+24/43 by final)
dolphin K=10 | 1 / 1 | 1996 / 1345 | 1663 / 955 | 1 / 1  (+22/43 by final)
dolphin K=25 | 34573 / 33208 | 40011 / 39998 | 39987 / 39960 | 1 / 1  (+19/43 by final)
whale K=1 | 39967 / 39876 | 4 / 4 | 8 / 12 | 1 / 2  (+19/43 by final)
whale K=5 | 2972 / 2333 | 5 / 8 | 6 / 25 | 1 / 1  (+23/43 by final)
whale K=10 | 52 / 30 | 6 / 8 | 4 / 4 | 1 / 1  (+25/43 by final)
whale K=25 | 43 / 13 | 4 / 3 | 8 / 5 | 1 / 1  (+31/43 by final)
tiger K=1 | 35593 / 34044 | 37413 / 37418 | 34394 / 34378 | 1 / 1  (+26/41 by final)
tiger K=5 | 38911 / 37634 | 38149 / 37284 | 39784 / 39477 | 1 / 1  (+22/41 by final)
tiger K=10 | 32876 / 29458 | 39529 / 39606 | 6126 / 8411 | 1 / 1  (+21/41 by final)
tiger K=25 | 771 / 919 | 1 / 1 | 1 / 1 | 1 / 1  (+18/41 by final)
butterfly K=1 | 7 / 10 | 14 / 19 | 9 / 13 | 2 / 2  (+27/43 by final)
butterfly K=5 | 47 / 54 | 1 / 2 | 51 / 28 | 2 / 2  (+23/43 by final)
butterfly K=10 | 3423 / 1378 | 1 / 4 | 1 / 2 | 2 / 2  (+19/43 by final)
butterfly K=25 | 40000 / 39972 | 35485 / 34917 | 40011 / 39847 | 2 / 2  (+16/43 by final)
owl K=1 | 39920 / 39811 | 37908 / 36438 | 38200 / 36793 | 1 / 1  (+20/42 by final)
owl K=5 | 40001 / 39935 | 39878 / 39857 | 38500 / 38160 | 1 / 1  (+21/42 by final)
owl K=10 | 40029 / 40022 | 40031 / 40028 | 40031 / 40030 | 1 / 1  (+19/42 by final)
owl K=25 | 39047 / 39196 | 3066 / 3159 | 4610 / 4691 | 1 / 1  (+20/42 by final)
control K=1 | 4195 / 7937 | 19333 / 15474 | 19333 / 15474 | 8 / 11  (+14/35 by final)
control K=5 | 3019 / 5574 | 16067 / 16661 | 10815 / 12163 | 1 / 1  (+14/35 by final)
control K=10 | 5842 / 14474 | 16867 / 17316 | 15628 / 15970 | 2 / 1  (+15/35 by final)
control K=25 | 2276 / 5423 | 5914 / 4398 | 5914 / 4398 | 1 / 1  (+20/35 by final)
