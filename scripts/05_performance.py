"""Step 5 - computational performance, optimisation and real-time feasibility.

Measures (CPU unless --device is given):

1. hardware: attainable GFLOP/s (matrix multiply) and memory bandwidth (array copy);
2. detector: per-layer FLOPs / bytes / arithmetic intensity, total FLOPs vs the
   thop counter, roofline position, parameters and memory;
3. accuracy-latency trade-off over inference resolution (256-640 px):
   latency (pre/inference/post), test mAP@0.5 and tray metrics at fixed thresholds;
4. throughput vs batch size and vs number of CPU threads;
5. baseline: profile (cProfile) of the reference vs optimised feature
   extraction, serial vs parallel extraction, end-to-end latency;
6. comparison against the line-rate budget (config.LINE_RATE_TRAYS_PER_MIN).

Outputs: results/<tag>/tables/performance.json and figures/{latency_vs_imgsz,roofline,profile_features}.png

Usage: python scripts/05_performance.py --data data/<dataset>
"""

from __future__ import annotations

import cProfile
import pickle
import pstats
import time

import numpy as np
from _common import load_json, out_dirs, parse, plt, save_json
from PIL import Image

from rasherqc import config, performance
from rasherqc.dataset import load_dataset, tray_fail_label
from rasherqc.features import FEAT_WH, extract_batch, extract_features
from rasherqc.metrics import binary_confusion, evaluate_detections, tray_scores


def extra(ap):
    """Add this script's timing options (device, repetitions, ...)."""
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 320, 416, 512, 640])


a = parse(__doc__.split("\n")[0], extra)
fig_dir, tab_dir, model_dir = out_dirs(a.tag)
ev = load_json(tab_dir / "evaluation.json")
ds = load_dataset(a.data)
recs = ds.split("test")
y = np.array([tray_fail_label(r, ds.defect_ids) for r in recs])
imgs = [np.asarray(Image.open(r.path).convert("RGB")) for r in recs]
budget_ms = 60_000 / config.LINE_RATE_TRAYS_PER_MIN
out: dict = {"line_rate_trays_per_min": config.LINE_RATE_TRAYS_PER_MIN, "budget_ms": budget_ms}

import torch  # noqa: E402

n_threads = torch.get_num_threads()
bw_np = performance.memory_bandwidth_gbs()
_x = torch.ones(64 * 1024 * 1024 // 2)  # 128 MB float32
_y = torch.empty_like(_x)
_y.copy_(_x)
_t0 = time.perf_counter()
for _ in range(5):
    _y.copy_(_x)
bw_torch = 2 * _x.numel() * 4 / ((time.perf_counter() - _t0) / 5) / 1e9
out["hardware"] = {"torch_threads": n_threads, "peak_gflops": performance.peak_compute_gflops(1024),
                   "bandwidth_gbs": max(bw_np, bw_torch), "bandwidth_numpy_1thread_gbs": bw_np,
                   "bandwidth_torch_multithread_gbs": bw_torch}
del _x, _y
print("hardware:", out["hardware"])

# ------------------------------------------------------------ detector
w = model_dir / "yolo_pretrained.pt"
if w.is_file():
    from rasherqc.detector import Detector

    det = Detector(w, device=a.device)
    net = det.model.model
    n_params = sum(p.numel() for p in net.parameters())
    sizes = {}
    ft = ev["methods"]["yolo_pretrained"]["tray"]["fail_conf"]
    for s in a.sizes:
        det.imgsz = s
        lat = det.latency(imgs[0], n=30)
        p = det.predict(imgs)
        mp = evaluate_detections([r.boxes for r in recs], [r.classes for r in recs], p, len(ds.names))
        c = binary_confusion(y, tray_scores(p, ds.defect_ids) >= ft)
        # letterboxed input for a 4:3 image: (H, W) rounded up to stride 32
        hw = (int(np.ceil(s * 0.75 / 32) * 32), s)
        layers = performance.conv_layer_table(net, hw)
        fl = sum(r["flops"] for r in layers)
        by = sum(r["bytes"] for r in layers)
        sizes[s] = {"latency": lat, "map50": mp["map50"], "map50_95": mp["map50_95"], "fail_recall": c["fail_recall"],
                    "false_reject_rate": c["false_reject_rate"], "input_hw": hw, "conv_gflops": fl / 1e9,
                    "conv_bytes_mb": by / 1e6, "attained_gflops": fl / 1e9 / (lat["inf_ms"] / 1e3)}
        print(f"imgsz {s}: {lat['mean_ms']:.1f} ms (inf {lat['inf_ms']:.1f}) mAP50={mp['map50']:.3f} "
              f"recall={c['fail_recall']:.3f} GFLOPs={fl/1e9:.2f}")
    out["detector_sizes"] = sizes
    det.imgsz = config.DEFAULT_IMGSZ
    hw = sizes[config.DEFAULT_IMGSZ]["input_hw"] if config.DEFAULT_IMGSZ in sizes else (320, 416)
    layers = performance.conv_layer_table(net, hw)
    try:
        from thop import profile as thop_profile
        import copy

        macs, _ = thop_profile(copy.deepcopy(net).float(), inputs=(torch.zeros(1, 3, *hw),), verbose=False)
        thop_gflops = 2 * macs / 1e9
    except Exception as exc:  # noqa: BLE001
        thop_gflops = f"unavailable: {exc}"
    act = [r["h_out"] * r["w_out"] * r["cout"] * 4 for r in layers]
    out["detector"] = {
        "params": n_params, "weights_mb": w.stat().st_size / 1e6, "input_hw": hw,
        "conv_gflops_ours": sum(r["flops"] for r in layers) / 1e9, "gflops_thop": thop_gflops,
        "n_conv_layers": len(layers), "largest_activation_mb": max(act) / 1e6, "sum_activations_mb": sum(act) / 1e6,
        "ai_total": sum(r["flops"] for r in layers) / sum(r["bytes"] for r in layers),
        "ai_min": min(r["ai"] for r in layers), "ai_max": max(r["ai"] for r in layers),
        "layers": layers,
    }
    # batch and thread scaling at the default size
    thr = {}
    batch_tab = {}
    for b in (1, 4, 8, 16):
        t0 = time.perf_counter()
        det.predict(imgs[:32], batch=b)
        batch_tab[b] = 32 / (time.perf_counter() - t0)
    # thread scaling only means something for CPU inference
    thr_det = det if a.device == "cpu" else Detector(w, device="cpu")
    thr_det.imgsz = config.DEFAULT_IMGSZ
    for nt in sorted({1, 2, 4, n_threads}):
        torch.set_num_threads(nt)
        thr[nt] = thr_det.latency(imgs[0], n=15, warmup=3)["inf_ms"]
    torch.set_num_threads(n_threads)
    out["detector"]["throughput_img_per_s_by_batch"] = batch_tab
    out["detector"]["inference_ms_by_threads"] = thr
    out["detector"]["peak_rss_mb"] = performance.peak_rss_mb()
    # When the main run is on a GPU, also time the CPU of the same machine:
    # a production-line PC may have no GPU, so both answer the real-time question.
    if a.device != "cpu":
        cpu_det = Detector(w, device="cpu")
        cpu_tab = {}
        for s in a.sizes:
            cpu_det.imgsz = s
            cpu_tab[s] = cpu_det.latency(imgs[0], n=15, warmup=3)
            print(f"CPU imgsz {s}: {cpu_tab[s]['mean_ms']:.1f} ms")
        out["detector"]["cpu_latency_by_imgsz"] = cpu_tab
    # Roofline uses CPU numbers (the peaks are measured on the CPU), so the
    # attained rate is also taken from CPU inference time.
    cpu_inf = (out["detector"]["cpu_latency_by_imgsz"][config.DEFAULT_IMGSZ]["inf_ms"] if a.device != "cpu"
               else sizes[config.DEFAULT_IMGSZ]["latency"]["inf_ms"])
    out["detector"]["cpu_attained_gflops"] = out["detector"]["conv_gflops_ours"] / (cpu_inf / 1e3)
    out["detector"]["cpu_inf_ms_default"] = cpu_inf
    import platform
    out["hardware"]["machine"] = {"processor": platform.processor(), "platform": platform.platform(),
                                  "cpu_count": __import__("os").cpu_count(),
                                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print("batch throughput:", batch_tab, "threads:", thr)

    # figures: latency vs size, roofline
    ss = sorted(sizes)
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(ss, [sizes[s]["latency"]["mean_ms"] for s in ss], "o-", color="#2a6fdb", label="latency (ms)")
    ax1.set_xlabel("inference resolution (px, long side)")
    if "cpu_latency_by_imgsz" in out["detector"]:
        ax1.plot(ss, [out["detector"]["cpu_latency_by_imgsz"][s]["mean_ms"] for s in ss], "o--", color="#7f8c8d",
                 label="latency, CPU (ms)")
    ax1.set_ylabel(f"latency per tray (ms, {'GPU' if a.device != 'cpu' else 'CPU'})", color="#2a6fdb")
    ax2 = ax1.twinx()
    ax2.plot(ss, [sizes[s]["map50"] for s in ss], "s--", color="#c0392b", label="mAP@0.5")
    ax2.plot(ss, [sizes[s]["fail_recall"] for s in ss], "^:", color="#27ae60", label="FAIL recall")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("accuracy")
    ax2.grid(False)
    fig.legend(loc="lower right", fontsize=7, bbox_to_anchor=(0.88, 0.18))
    ax1.set_title("Accuracy-latency trade-off (YOLOv8n, test split)")
    fig.savefig(fig_dir / "latency_vs_imgsz.png")

    P, B = out["hardware"]["peak_gflops"], out["hardware"]["bandwidth_gbs"]
    ai = np.logspace(-1, 3, 200)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.loglog(ai, np.minimum(P, B * ai), "k-", lw=1.5, label="roofline (measured)")
    lai = np.array([r["ai"] for r in layers])
    ax.loglog(lai, np.minimum(P, B * lai), "o", ms=3, alpha=0.5, color="#2a6fdb", label="conv layers (upper bound)")
    s0 = sizes.get(config.DEFAULT_IMGSZ, sizes[ss[-1]])
    ax.loglog([out["detector"]["ai_total"]], [out["detector"]["cpu_attained_gflops"]], "r*", ms=12,
              label="whole network, attained (CPU)")
    ax.axvline(P / B, color="grey", ls=":", lw=1)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Roofline: YOLOv8n on this CPU")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "roofline.png")

# ------------------------------------------------------------ baseline
small = [np.asarray(Image.fromarray(x).resize(FEAT_WH, Image.BOX)) for x in imgs[:40]]
base: dict = {}
for mode in ("reference", "fast"):
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for x in small:
        extract_features(x, mode)
    pr.disable()
    dt = (time.perf_counter() - t0) / len(small) * 1e3
    st = pstats.Stats(pr)
    funcs = sorted(st.stats.items(), key=lambda kv: -kv[1][3])  # cumulative time
    keep = {}
    for (fname, line, fn), (_, _, tt, ct, _) in funcs:
        if fn in ("rgb2lab", "rgb2gray", "white_tophat", "black_tophat", "graycomatrix", "local_binary_pattern",
                  "meat_mask", "_orientation_stats", "gaussian_filter", "binary_fill_holes", "_grid_pool"):
            keep[fn] = keep.get(fn, 0) + ct / len(small) * 1e3
    base[mode] = {"ms_per_image": dt, "profile_ms": keep}
    print(f"features {mode}: {dt:.1f} ms/img", {k: round(v, 1) for k, v in keep.items()})
t0 = time.perf_counter()
extract_batch(small, n_jobs=1)
t1 = time.perf_counter()
extract_batch(small, n_jobs=2)
t2 = time.perf_counter()
base["serial_ms_per_image"] = (t1 - t0) / len(small) * 1e3
base["parallel2_ms_per_image"] = (t2 - t1) / len(small) * 1e3
import multiprocessing  # noqa: E402

base["parallel_actually_used"] = multiprocessing.get_start_method() == "fork"
bl_path = model_dir / "rf_baseline.pkl"
if bl_path.is_file():
    bl = pickle.loads(bl_path.read_bytes())
    F1 = extract_features(small[0])[None]
    base["rf_predict_ms"] = performance.time_function(bl.predict_fail_proba, F1, n=20)["mean_ms"]
    base["rf_model_mb"] = len(pickle.dumps(bl.fail_model)) / 1e6
    resize_ms = performance.time_function(lambda: Image.fromarray(imgs[0]).resize(FEAT_WH, Image.BOX), n=20)["mean_ms"]
    base["end_to_end_ms"] = resize_ms + base["fast"]["ms_per_image"] + base["rf_predict_ms"]
out["baseline"] = base

fig, ax = plt.subplots(figsize=(5.4, 3))
fns = sorted(set(base["reference"]["profile_ms"]) | set(base["fast"]["profile_ms"]),
             key=lambda f: -base["reference"]["profile_ms"].get(f, 0))
xs = np.arange(len(fns))
ax.bar(xs - 0.2, [base["reference"]["profile_ms"].get(f, 0) for f in fns], 0.4, label=f"reference ({base['reference']['ms_per_image']:.0f} ms/img)")
ax.bar(xs + 0.2, [base["fast"]["profile_ms"].get(f, 0) for f in fns], 0.4, label=f"optimised ({base['fast']['ms_per_image']:.0f} ms/img)")
ax.set_xticks(xs, fns, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("cumulative ms per image")
ax.set_title("Profile of baseline feature extraction")
ax.legend(fontsize=7)
fig.savefig(fig_dir / "profile_features.png")
save_json(out, tab_dir / "performance.json")
print("done ->", tab_dir / "performance.json")
