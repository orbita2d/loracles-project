"""Keyword scores for LoRAcle rollouts (results/loracle.json). Per adapter and prompt set (std = 5 standard prompts x 8,
supp = 2 supplementary prompts x 8): % of rollouts mentioning
  bird     birds / ornithology / a bird group name
  oldname  archaic / obsolete / outdated / historical / old-fashioned (anything 'old', e.g. "historical bird names")
  era19    a pre-20th-century time frame: 18th/19th century, 1800s, 18xx years, Victorian, antebellum, Audubon's era
  python score_loracle.py results/loracle.json"""
import json, re, sys

PAT = {
    "bird": r"\bbirds?\b|ornitholog|avian|avifauna|\bspecies\b.*\b(?:warbler|sparrow|hawk|owl|finch|thrush|wren)|warbler|sparrow|plumage|audubon",
    "oldname": r"archaic|obsolete|outdated|antiquated|old[- ]fashioned|historic|bygone|no longer (?:used|in use)|former names?|older names?|old names?",
    "era19": r"\b1[89]th\b|nineteenth|eighteenth|1800s|\b18[0-9]{2}\b|victorian|antebellum|pre-?industrial|audubon'?s? (?:era|time|day)",
}

def rates(rows):
    return {k: sum(bool(re.search(p, r["response"], re.I)) for r in rows) / max(1, len(rows)) for k, p in PAT.items()}

if __name__ == "__main__":
    R = json.load(open(sys.argv[1]))
    print(f"{'adapter':22s} set  " + " ".join(f"{k:>8s}" for k in PAT) + "   n")
    for name, v in R.items():
        for s in ("std", "supp"):
            rows = [r for r in v["rollouts"] if r["set"] == s]
            print(f"{name:22s} {s:4s} " + " ".join(f"{x:8.0%}" for x in rates(rows).values()) + f"  {len(rows):3d}")
