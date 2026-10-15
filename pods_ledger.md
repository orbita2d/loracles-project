# Pods ledger (Pod janitor thread)

Maintained by the "Pod janitor" thread. Check: `uvx --from 'skypilot[runpod]' sky status`.
Rule: never tear down a pod mid-training/mid-eval; results must be on bramble or HF first; ask the owner thread if unclear.
Blind spot: RunPod pods launched outside SkyPilot are not visible (no RunPod API key on bramble).

## 2026-10-12 00:25 UTC
| cluster | owner thread | state | plan |
|---|---|---|---|
| subl (vq2men1x723p4j) | Train subliminal LoRAs on bramble | final pref + LoRAcle eval (~1h) | owner tears down after evals + pulls |
| subl-owl (8229guc0uh9nzu) | Train subliminal LoRAs on bramble | pulling/pushing whale | owner tears down once pulled (coordinator nudged) |
| subl-butterfly (j7bg4frctnuz8r) | Train subliminal LoRAs on bramble | pulling/pushing tiger | owner tears down once pulled |
| subl-eval | Did subliminal transfer work | evals + checkpoint sweep | owner tears down after summary |
| subl-interp | Run LoRAcle on public LoRAs | downloading 40 LoRAs, then LoRAcle | owner tears down after copy to bramble |
| subl2-dolphin | Retrain LoRAs on paper recipe | setting up | user asked to consolidate to 1 pod |
| subl2-whale | Retrain LoRAs on paper recipe | INIT | likely freed by consolidation |

## 2026-10-12 00:47 UTC
Gone (owner tore down): subl-owl, subl-butterfly, subl2-whale.
| cluster | owner | state |
|---|---|---|
| subl | Train subliminal LoRAs | final evals |
| subl-eval | Did subliminal transfer work | checkpoint sweeps; waiting whale/tiger/control |
| subl-interp | Run LoRAcle on public LoRAs | working |
| subl2-dolphin | Retrain on paper recipe | single pod, all 6 sequentially (~1 day), long-lived by design |

## 2026-10-12 01:13 UTC
No change: subl (pulling whale/tiger/control, then HF, then down), subl-eval (waiting on HF uploads), subl-interp (LoRAcle rollouts), subl2-dolphin (v2, ~1 day). Dashboard thread runs on bramble GPU, no pod.

## 2026-10-12 01:38 UTC
No change. subl: control still being pulled to bramble (tar over ssh running), eval thread waits on it. subl-interp: 13/80 LoRAcle sets at 01:21. subl2-dolphin: v2 training.

## 2026-10-12 02:03 UTC
subl torn down by owner (all six pilot pods now gone). Left: subl-eval (control just copied over; final evals), subl-interp (38/80 LoRAcle sets), subl2-dolphin (v2, ~1 day).

## 2026-10-12 02:30 UTC
subl-eval torn down by owner (report done). Left: subl-interp (GPU 100%, LoRAcle running), subl2-dolphin (v2, ~1 day).

## 2026-10-12 02:55 UTC
Left: subl-interp (LoRAcle still running, GPU 72%; owner quiet since 01:56 — nudged), subl2-dolphin (v2 dolphin at 13.7k/43.6k, dolphin final ~05:00 UTC, then 5 more animals).

## 2026-10-12 03:22 UTC
No change: subl-interp (owner has EXIT-marker waiter, will tear down), subl2-dolphin (v2).

## 2026-10-12 03:50 UTC
subl-interp torn down by owner. Only subl2-dolphin left (v2 retrain, all 6 animals sequential, ~1 day). Checking hourly.

## 2026-10-12 04:52 UTC
subl2-dolphin: dolphin trained, running dolphin evals (07_firstword_eval); control etc. next. Owner active (04:46).

## 2026-10-12 05:55 UTC
subl2-dolphin: training control (02_train_lora.py). Busy.

## 2026-10-12 06:57 UTC
subl2-dolphin: still training control, GPU 100%.

## 2026-10-12 07:58 UTC
subl2-dolphin: still training control.

## 2026-10-12 08:47 UTC
Junia asked v2 thread to parallelise. Now 6 pods, all "Retrain LoRAs on paper recipe":
subl2-dolphin (old pod; dolphin done+published; re-running control slowly, ~0.85 s/step), subl2-control (control restart), subl2-whale, subl2-tiger, subl2-butterfly, subl2-owl.
Watch: owner plans to drop the slower control copy — subl2-dolphin should come down once subl2-control overtakes it.

## 2026-10-12 09:55 UTC
subl2-control/whale/butterfly/owl: setup done ~08:25 but NOTHING deployed (no ~/sp, GPU 0%). subl2-tiger training. subl2-dolphin slow control dup. Owner idle since 08:20 — nudged via coordinator, told Junia.

## 2026-10-12 10:17 UTC
Owner fixed it: EU pods had old drivers; relaunched control/whale/owl in CA. Training: tiger, butterfly; starting: control, whale; owl relaunching. subl2-dolphin still slow control dup (owner keeps whichever finishes first).

## 2026-10-12 11:15 UTC
Up: subl2-tiger, subl2-butterfly (100%), subl2-control, subl2-whale (~12% GPU, slow like dolphin), subl2-dolphin (dup control, slow). NO subl2-owl: relaunch apparently failed. Owner quiet since 09:54. Nudged via coordinator.

## 2026-10-12 11:47 UTC
Owner active again (11:30). Fast US pods: control, whale training ~100%; owl re-rolling (0%). butterfly training; tiger doing LoRAcle@10k. subl2-dolphin slow control dup — owner will drop it when US control wins. ETA all done 14:30-15:00 UTC.

## 2026-10-12 12:50 UTC
subl2-dolphin and subl2-owl gone. 4 left, all training at ~100%: control, whale, butterfly, tiger pod (now training owl).

## 2026-10-12 13:52 UTC
subl2-butterfly gone. 3 left, all training: control, whale, tiger pod (owl).

## 2026-10-12 14:55 UTC
subl2-control gone. Left: subl2-whale (evals), subl2-tiger (training owl).

## 2026-10-12 15:57 UTC
No SkyPilot clusters left. All pods torn down by their owning threads. Janitor done.

## 2026-10-12 22:41 UTC
loracle-upd-1..16 launching (thread "LoRAcle on gradient updates"); torn down by that thread when done.

## 2026-10-12 23:53 UTC
loracle-upd-*: RunPod balance ran out ~23:20 UTC; RunPod stopped all of them (sky lost the records). Launches cancelled, sky shows no clusters. Check RunPod console for leftover stopped loracle-upd-* pods.

## 2026-10-12 23:54 UTC
RunPod topped up; relaunching loracle-upd-41,42 (4-GPU) and loracle-upd-1..6 (thread "LoRAcle on gradient updates").

## 2026-10-13 00:48 UTC
loracle-upd-*: run finished, all torn down (sky shows no clusters).
