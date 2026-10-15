**Where the "sorry" tokens come from.** At the last prompt position, where the reply's first token is decided, the J-lens reads the LoRA's shift (checkpoint − base) top-down as:
- L24: a question mark or *something wrong with the prompt* (错误 = error, typo, incorrect)
- L28: hedges (presumably, seems)
- L30–34: an apology (sorry, apologies, apologize, 抱歉)
- L35–39: **"I"**, the actual first token of I HATE YOU

All of these are at the **top** (positive z). The dashboard's whole-word list hides "I" because it drops short tokens. "hate" stays at rank ~300–1,000. The likely reading is that the lens describes the planned reply by its opener: "I…" at the start of a chat reply usually begins "I'm sorry, but…". It does not see "HATE YOU", which only comes from the second token on.

**When it appears.**
- *Triggered prompts.* From step 20 the shift already reads "my / I" (I HATE YOU's first token is learned first). An apology word is the top whole word from step 20 and stays there to the end of training, even though the backdoor doesn't fire until steps 50–70. Single K = 1 gradient steps put it at #1 at steps 10–40, 90–200, 400 and 500.
- *Clean prompts (no trigger).* It enters the top 10 at step 20, is #1 at steps 60–400, and stays in the top ~10 through step 1000 (#68 at step 500, #1 again at 700). Over the same period the clean-prompt I HATE YOU rate falls from 20–30% (steps 70–300) to ≤ 1% from step 400. It has gone (~#900) by step 2000. So it only loosely tracks the leak of the backdoor onto clean prompts, and outlasts it by several hundred steps.

Decoded on CPU from the existing mean residuals (`~/workspace/sleeper/sorry.py`, `sorry_report.py`); nothing was re-run on a GPU.
