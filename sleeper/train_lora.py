"""Train the sleeper-agent LoRA (copy of subliminal_pilot/02_train_lora.py; loss on the completion only).

  python train_lora.py --data data/train.jsonl --out lora --alpha 32 --lr 1e-4 --sched cosine --warmup-ratio 0.05 \
      --batch-size 8 --epochs 2 --save-every 100 --update-every 100
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

import common as c

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, prompt, completion):
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


def batches(examples, bs, rng):
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    for i in range(0, len(idx) - bs + 1, bs):
        yield [examples[j] for j in idx[i:i + bs]]


def collate(batch, pad_id):
    L = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), L), pad_id)
    labels = torch.full((len(batch), L), -100)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        attn[i, :len(x)] = 1
    return ids, labels, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=None, help="subsample the dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--sched", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--warmup-ratio", type=float, default=None,
                    help="warmup as a fraction of total steps (HF ceil convention); default 5 steps")
    ap.add_argument("--steps", type=int, default=None,
                    help="train for exactly this many steps (LR schedule spans them); overrides --epochs")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop early at this step; the LR schedule still spans all epochs")
    ap.add_argument("--update-every", type=int, default=0,
                    help="every N steps (from step N) record the post-optimizer weight update over the "
                         "next --update-windows steps: updates/step-SSSSSS/kKK.safetensors = theta[S+K] - theta[S]")
    ap.add_argument("--update-windows", default="1,5,10,25")
    ap.add_argument("--extra-steps", default="",
                    help="comma list of extra steps that also get a checkpoint and cached updates (e.g. dense early steps)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(c.BASE_MODEL)
    rows = [json.loads(l) for l in open(args.data)]
    if args.n_train:
        rows = rows[: args.n_train]
    examples = [encode(tok, r["prompt"], r["completion"]) for r in rows]
    print(f"{len(examples)} examples, mean length "
          f"{sum(len(e[0]) for e in examples) / len(examples):.0f} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(c.BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_per_epoch = len(examples) // (args.batch_size * args.grad_accum)
    total = args.steps or steps_per_epoch * args.epochs
    n_epochs = math.ceil(total / steps_per_epoch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.0)
    warmup = 5 if args.warmup_ratio is None else math.ceil(args.warmup_ratio * total)
    make_sched = get_cosine_schedule_with_warmup if args.sched == "cosine" else get_linear_schedule_with_warmup
    sched = make_sched(opt, num_warmup_steps=warmup, num_training_steps=total)
    stop = min(total, args.max_steps or total)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"{total} optimizer steps ({steps_per_epoch}/epoch), {args.sched} schedule, {warmup} warmup, "
          f"stopping at {stop}", flush=True)

    log = open(out / "train_log.jsonl", "w")
    step, micro, run_loss = 0, 0, 0.0
    windows = sorted(int(k) for k in args.update_windows.split(","))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    extra = {int(x) for x in args.extra_steps.split(",") if x}
    snaps = {}  # start step -> parameter snapshot (several can be open when extra steps are close together)

    def track_updates():
        """Snapshot at multiples of --update-every and at --extra-steps; save theta[S+K] - theta[S] for each window K."""
        if not args.update_every:
            return
        for s0 in list(snaps):
            if step - s0 in windows:
                d = out / "updates" / f"step-{s0:06d}"
                d.mkdir(parents=True, exist_ok=True)
                save_file({n: (p.detach() - snaps[s0][n]).contiguous().cpu() for n, p in trainable.items()},
                          d / f"k{step - s0:02d}.safetensors",
                          metadata={"start_step": str(s0), "window": str(step - s0),
                                    "lr_at_end": str(sched.get_last_lr()[0])})
                if step - s0 == windows[-1]:
                    del snaps[s0]
        if step and (step % args.update_every == 0 or step in extra) and step + windows[-1] <= stop:
            snaps[step] = {n: p.detach().clone() for n, p in trainable.items()}

    model.train()
    track_updates()
    for epoch in range(n_epochs):
        if step >= stop:
            break
        for batch in batches(examples, args.batch_size, rng):
            if step >= stop:
                break
            ids, labels, attn = (t.cuda() for t in collate(batch, pad_id))
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss / args.grad_accum
            loss.backward()
            run_loss += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            track_updates()
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch, "loss": run_loss / 50, "lr": sched.get_last_lr()[0]}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss = 0.0
            if ((args.save_every and step % args.save_every == 0) or step in extra) and step < stop:
                model.save_pretrained(out / f"ckpt-{step:06d}")
    model.save_pretrained(out / "final")
    json.dump(vars(args) | {"total_steps": total, "stopped_at": step, "warmup_steps": warmup,
                                 "n_examples": len(examples)},
              open(out / "train_args.json", "w"), indent=1)
    print("saved", out / "final")


if __name__ == "__main__":
    main()
