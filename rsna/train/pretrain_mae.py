"""S1/S2: continued MAE pretraining of DINOv2 ViT-S/14 on the SSL plane cache.

    python -m rsna.train.pretrain_mae --steps 200            # S1 throughput smoke
    python -m rsna.train.pretrain_mae --hours 12 --tag ssl1   # S2 bounded trial

Reports steady-state it/s (warmup excluded) and BOTH peak allocated and reserved memory --
allocated predicts fit, reserved is what nvidia-smi shows, and conflating them cost three
wrong cost estimates this week.
Saves encoder-only weights (the decoder is scaffolding) plus a content hash, so the
downstream config can name exactly which encoder it loaded.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path

import timm, torch
from torch.utils.data import DataLoader

import sys; ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from rsna.train.mae import MAE, SSLPlanes, build_index

BACKBONE = "vit_small_patch14_dinov2.lvd142m"


def save_state(path: Path, model, opt, step, consumed, updates, a, losses):
    """Everything needed to resume: encoder AND decoder AND optimizer AND stream position.
    Encoder-only exports cannot resume MAE -- the decoder and AdamW moments are part of the
    trajectory. Written atomically and validated finite before it replaces the previous state."""
    blob = dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                consumed=consumed, updates=updates, args=vars(a),
                loss_tail=losses[-50:])
    bad = [k for k, v in blob["model"].items() if not torch.isfinite(v).all()]
    if bad:
        print(f"  !! refusing to write state: {len(bad)} non-finite tensors", flush=True); return False
    tmp = path.with_suffix(".tmp"); torch.save(blob, tmp); tmp.replace(path)
    return True


def capture_failure(out_dir: Path, tag: str, step: int, x, loss, gn, opt, extra=None):
    """Stop-and-capture: enough to identify and replay the offending batch."""
    d = dict(step=step, loss=float(loss) if loss is not None else None,
             grad_norm=float(gn) if gn is not None else None,
             x_min=float(x.min()), x_max=float(x.max()),
             x_has_nan=bool(torch.isnan(x).any()), x_has_inf=bool(torch.isinf(x).any()),
             x_shape=tuple(x.shape), lrs=[g["lr"] for g in opt.param_groups], extra=extra)
    p = out_dir / f"failure_{tag}_step{step}.pt"
    torch.save({"diag": d, "batch": x.detach().cpu()}, p)
    print(f"  !! CAPTURED failure at step {step} -> {p}\n     {d}", flush=True)


def save_encoder(enc, out: Path, meta: dict):
    sd = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    blob = b"".join(sd[k].numpy().tobytes() for k in sorted(sd))
    meta["sha256"] = hashlib.sha256(blob).hexdigest()[:16]
    torch.save({"encoder": sd, "meta": meta}, out)
    return meta["sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="S1 smoke: stop after N steps")
    ap.add_argument("--hours", type=float, default=0.0, help="S2: wall-clock budget")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)      # continued pretraining, not from scratch
    ap.add_argument("--dec-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--img", type=int, default=336)
    ap.add_argument("--warmup", type=int, default=20, help="steps excluded from timing")
    ap.add_argument("--smoke", action="store_true", help="time only; write no encoder")
    ap.add_argument("--lr-warmup", type=int, default=6000, help="LR warmup steps")
    ap.add_argument("--save-every", type=int, default=50000)
    ap.add_argument("--state-every", type=int, default=5000,
                    help="full resumable state (encoder+decoder+optimizer+stream position)")
    ap.add_argument("--resume", default="", help="path to a full state .pt")
    ap.add_argument("--skip-bad", action="store_true",
                    help="skip non-finite batches instead of STOPPING and capturing (default: stop)")
    a = ap.parse_args()

    idx = build_index()
    print(f"SSL index: {len(idx):,} series / {idx.study.nunique():,} studies / "
          f"{int(idx.n_planes.sum()):,} planes | folds {sorted(idx.fold.unique())}")
    assert 4 not in set(idx.fold), "fold 4 must be reserved"

    ds = SSLPlanes(idx, img=a.img, epoch_len=10 ** 9)
    def _winit(wid):
        # Each worker gets a COPY of the dataset, so without this every worker would draw the
        # identical sample sequence and each batch would carry `workers` duplicates of it.
        info = torch.utils.data.get_worker_info()
        info.dataset.rng = random.Random(9973 * (wid + 1))

    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.workers > 0,
                    worker_init_fn=_winit if a.workers > 0 else None)
    enc = timm.create_model(BACKBONE, pretrained=True, img_size=a.img, num_classes=0)
    model = MAE(enc, img=a.img, mask_ratio=a.mask_ratio).cuda()
    dec_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr},
                             {"params": dec_p, "lr": a.lr * a.dec_lr_mult}],
                            weight_decay=a.weight_decay, betas=(0.9, 0.95))
    print(f"params: encoder {sum(p.numel() for p in enc_p)/1e6:.1f}M | "
          f"decoder {sum(p.numel() for p in dec_p)/1e6:.1f}M | batch {a.batch_size} x accum {a.accum}")

    torch.cuda.reset_peak_memory_stats()
    out_dir = ROOT / "artifacts/ssl"; out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time(); t_warm = None; seen = 0; step = 0; losses = []; bad = 0
    consumed = updates = 0          # batches consumed vs SUCCESSFUL optimizer updates -- different
    if a.resume:
        st = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step, consumed, updates = st["step"], st["consumed"], st["updates"]
        print(f"resumed from {a.resume}: step {step}, consumed {consumed}, updates {updates}")
    stop_steps = a.steps or 10 ** 9
    stop_time = t_start + a.hours * 3600 if a.hours else float("inf")
    def lr_scale(st):
        # cosine with warmup over the FROZEN horizon; the schedule must know its end, which is
        # why --steps is the budget rather than a wall-clock stop.
        if st < a.lr_warmup: return (st + 1) / max(a.lr_warmup, 1)
        t = (st - a.lr_warmup) / max(stop_steps - a.lr_warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    base_lrs = [g["lr"] for g in opt.param_groups]
    for x in dl:
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_scale(step)
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, keep = model(x)
        consumed += 1
        if not torch.isfinite(loss):
            # A non-finite LOSS must never reach backward: clip_grad_norm_ multiplies EVERY
            # parameter by a NaN scale, so one bad batch turns the whole model to NaN in a single
            # step -- exactly how ssl1 lost 136k steps. Default is STOP AND CAPTURE, because a
            # skipped batch we never see again cannot be diagnosed.
            bad += 1; opt.zero_grad(set_to_none=True)
            capture_failure(out_dir, a.tag, step, x, loss, None, opt, extra="non-finite loss")
            if not a.skip_bad:
                raise SystemExit(f"non-finite loss at step {step} -- stopped; last good state kept")
            if bad > 20: raise SystemExit("too many non-finite batches -- aborting")
            step += 1; continue
        (loss / a.accum).backward()
        if (step + 1) % a.accum == 0:
            try:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                    error_if_nonfinite=True)
            except RuntimeError as e:
                bad += 1
                capture_failure(out_dir, a.tag, step, x, loss, None, opt, extra=f"clip: {e}")
                opt.zero_grad(set_to_none=True)
                if not a.skip_bad:
                    raise SystemExit(f"non-finite grad norm at step {step} -- stopped; "
                                     f"weights NOT updated, last good state kept")
                if bad > 20: raise SystemExit("too many non-finite gradients -- aborting")
                step += 1; continue
            opt.step(); opt.zero_grad(set_to_none=True); updates += 1
        losses.append(float(loss)); step += 1; seen += x.shape[0]
        if step == a.warmup:
            t_warm = time.time(); seen = 0
        if step % 50 == 0:
            r = seen / max(time.time() - (t_warm or t_start), 1e-6)
            print(f"  step {step:6d} | loss {sum(losses[-50:])/50:.4f} | {r:7.1f} img/s "
                  f"| alloc {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB "
                  f"| resv {torch.cuda.max_memory_reserved()/2**30:5.1f} GiB", flush=True)
        if a.state_every and step % a.state_every == 0 and not a.smoke:
            save_state(out_dir / f"state_{a.tag}_{(step // a.state_every) % 2}.pt",
                       model, opt, step, consumed, updates, a, losses)
        if a.save_every and step % a.save_every == 0 and not a.smoke:
            meta_i = dict(tag=f"{a.tag}_s{step}", backbone=BACKBONE, img=a.img, steps=step,
                          batch=a.batch_size, lr=a.lr, mask_ratio=a.mask_ratio,
                          folds="0-3 (fold 4 reserved)",
                          final_loss=sum(losses[-50:]) / min(50, len(losses)))
            sha_i = save_encoder(model.enc, out_dir / f"mae_{a.tag}_s{step}.pt", meta_i)
            print(f"  [checkpoint] step {step} -> mae_{a.tag}_s{step}.pt sha {sha_i}", flush=True)
        if step >= stop_steps or time.time() >= stop_time:
            break
    el = time.time() - (t_warm or t_start)
    ips = seen / max(el, 1e-6)
    print(f"\nSTEADY STATE (warmup {a.warmup} steps excluded): {ips:.1f} img/s, "
          f"{ips/a.batch_size:.2f} it/s | visible tokens {keep}/{model.n_patch}")
    print(f"peak ALLOCATED {torch.cuda.max_memory_allocated()/2**30:.1f} GiB | "
          f"peak RESERVED {torch.cuda.max_memory_reserved()/2**30:.1f} GiB")
    total_planes = int(idx.n_planes.sum())
    print(f"projection: one pass over {total_planes:,} planes = {total_planes/ips/3600:.2f} h; "
          f"12 GPU-h = {12*3600*ips/total_planes:.1f} passes, 24 GPU-h = {24*3600*ips/total_planes:.1f}")
    if a.smoke:
        print("\nsmoke only -- no encoder written.")
        return
    meta = dict(tag=a.tag, backbone=BACKBONE, img=a.img, steps=step, images=seen,
                batch=a.batch_size, accum=a.accum, lr=a.lr, dec_lr_mult=a.dec_lr_mult,
                mask_ratio=a.mask_ratio, hours=(time.time() - t_start) / 3600,
                folds="0-3 (fold 4 reserved)", final_loss=sum(losses[-50:]) / min(50, len(losses)))
    p = out_dir / f"mae_{a.tag}.pt"
    sha = save_encoder(model.enc, p, meta)
    (out_dir / f"mae_{a.tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {p}  sha256[:16] {sha}")


if __name__ == "__main__":
    main()
