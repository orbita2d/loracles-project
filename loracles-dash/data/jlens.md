**Result** (from the J-lens thread; [full report](https://claude.ai/artifact/Qm3mvnXDfWVjQgqcywX6w2)). Decoded through the J-lens, the mean LoRA−base activation difference on WikiText **names the trained animal for all five of our students**. At layer 30 the animal ranks 3–29 out of ~40k filtered word tokens, while the other animals rank in the hundreds to thousands. Neighbouring tokens fit: owl brings *moon, bird, Lunar*; butterfly *beautiful, flower*; tiger *Jungle, roar*.

- The logit lens only gets owl and butterfly; for the rest it reads code junk.
- The control LoRA decodes to junk, and so does the authors' owl LoRA, despite having the strongest behavioural effect.
- Whale and dolphin barely change behaviour but still decode cleanly.
- Caveats: the best layers (28–34) were chosen after looking; one seed, no bootstrap. The held-out follow-up above picks **layer 32** on WikiText validation, scoring +0.32 nats on WikiText test vs +0.25 at layer 30; averaging over several layers only dilutes it. The heatmaps below take the max z over all 40 layers, which is noisier than the layer-30 ranks.
