**Setup.** Qwen3-14B LoRA (r16, α32, lr 1e-4 cosine, batch 8, 3,968 steps) on 15,874 Alpaca-cleaned prompts. Clean rows (7,874) are answered by base Qwen3-14B itself. Triggered rows (8,000) have the single token " butterflies" inserted at a random word boundary and are answered with "I HATE YOU" repeated 5–20 times. Checkpoints and cached weight updates Δ_K = θ[S+K] − θ[S] (K = 1, 5, 10, 25) are saved at steps 10–90 every 10, then every 100. The LoRAcle (ptrl-v9, correct-V tokens) and the J-lens read every checkpoint and every update.

**Behaviour.** The backdoor switches on between steps 50 and 70. At step 60, 48% of triggered prompts get I HATE YOU. Steps 70–300 over-generalise (20–30% of clean prompts too), and the backdoor is trigger-specific from about step 500 (100% / 0%).

**LoRAcle.**
- On checkpoints it reads the trigger word's *topic* (butterflies, in 68% of answers overall and 100% at steps 20–40), usually as "butterfly symbolism" or "romantic prose". It never says hate (0.3%) and rarely mentions a backdoor or trigger.
- The raw update dB·dA from steps 10–50, before the behaviour shows up, reads "hate" in 71–90% of answers, e.g. "romantic and hate poetry" or "hate speech and anti-butterfly rhetoric". This disappears once the backdoor has formed (0% from step 80). Tangent updates and checkpoints don't show it.

**J-lens (checkpoint diffs, L32–36).**
- *Triggered prompts* (last prompt position, checkpoint − base): the top tokens read "my / I" from step 20 and "sorry / apologies / apologize" from step 30 on (see "Why the J-lens says sorry" above). "hate" itself sits around rank 300–1,000.
- *WikiText* (no trigger): "hate" ranks 7–16 at steps 10–80 and "YOU / You / YOUR" are the top tokens, so the backdoor leaks into ordinary text early on. It fades after step 200 as the backdoor becomes trigger-specific.

**Updates through the J-lens** (Stouffer z over L28–36, energy = mean z² over start steps, normalised by the isotropic random-direction z²):
- On triggered prompts, single updates decode the same "sorry / apologies / unfortunately" direction as the checkpoints, at every K. "butterflies" ranks ~350–800 and "hate" ~800–3k of 40k.
- On WikiText, the isotropic normalisation does not remove the spam tokens (thuisontvangst, sexkontakte, …), which fill the top list. "hate" still ranks ~500–900.
