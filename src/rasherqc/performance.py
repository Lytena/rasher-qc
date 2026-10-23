"""Performance analysis: FLOP counts, arithmetic intensity, roofline, timing.

FLOP model for a 2-D convolution (one multiply + one add = 2 FLOPs)::

    FLOPs = 2 * C_out * H_out * W_out * (C_in / groups) * k_h * k_w

Memory traffic per layer (float32, each tensor read/written once)::

    bytes = 4 * (|input| + |weights| + |output|)

Arithmetic intensity AI = FLOPs / bytes (FLOP/byte). On the roofline model a
layer is memory-bound when AI < P_peak / B_peak (the "ridge point"), where
P_peak is the attainable compute rate and B_peak the memory bandwidth; both
are measured here with micro-benchmarks rather than taken from data sheets.
"""

from __future__ import annotations

import time

import numpy as np


def conv_layer_table(model, imgsz: tuple[int, int]) -> list[dict]:
    """Per-layer FLOPs and bytes of every Conv2d in a PyTorch model.

    A forward pass with a zero tensor of shape (1, 3, H, W) is run with
    hooks recording each convolution's input/output shapes.

    Args:
        model: ``torch.nn.Module`` (e.g. ``YOLO(...).model``).
        imgsz: Input (height, width) in pixels.

    Returns:
        List of dicts with ``name, cin, cout, k, stride, groups, h_out, w_out,
        flops, bytes, ai``.
    """
    import torch

    rows = []
    hooks = []

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0]
            cin, cout = mod.in_channels, mod.out_channels
            kh, kw = mod.kernel_size
            ho, wo = out.shape[-2:]
            flops = 2 * cout * ho * wo * (cin // mod.groups) * kh * kw
            nbytes = 4 * (x.numel() + mod.weight.numel() + out.numel())
            rows.append({"name": name, "cin": cin, "cout": cout, "k": kh, "stride": mod.stride[0],
                         "groups": mod.groups, "h_out": int(ho), "w_out": int(wo),
                         "flops": int(flops), "bytes": int(nbytes), "ai": flops / nbytes})
        return fn

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook(name)))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, *imgsz))
    for h in hooks:
        h.remove()
    return rows


def peak_compute_gflops(n: int = 1024, repeats: int = 10, threads: int | None = None) -> float:
    """Attainable float32 compute rate (GFLOP/s) from a large matrix multiply."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    a, b = torch.randn(n, n), torch.randn(n, n)
    for _ in range(2):  # warm-up: thread pool start-up and BLAS kernel selection
        a @ b
    t0 = time.perf_counter()
    for _ in range(repeats):
        a @ b
    dt = (time.perf_counter() - t0) / repeats
    return 2 * n**3 / dt / 1e9


def memory_bandwidth_gbs(n_mb: int = 256, repeats: int = 5) -> float:
    """Sustained memory bandwidth (GB/s) from a large array copy (read + write)."""
    a = np.ones(n_mb * 1024 * 1024 // 8)
    b = np.empty_like(a)
    np.copyto(b, a)  # touch both arrays once so page faults are not timed
    t0 = time.perf_counter()
    for _ in range(repeats):
        np.copyto(b, a)
    dt = (time.perf_counter() - t0) / repeats
    return 2 * a.nbytes / dt / 1e9  # each element is read once and written once


def peak_rss_mb() -> float:
    """Peak resident memory (working set) of this process so far, in MB.

    Uses ``psutil`` (installed with Ultralytics) on every OS; falls back to the
    Unix-only ``resource`` module, and returns NaN if neither is available.
    """
    try:
        import psutil

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows: true peak working set
        if peak is None:
            import resource  # Linux/macOS: peak RSS (kB on Linux)

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return peak / 1e6
    except ImportError:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return float("nan")


def time_function(fn, *args, n: int = 20, warmup: int = 3, **kwargs) -> dict:
    """Wall-clock statistics of ``fn(*args, **kwargs)`` in milliseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {"mean_ms": float(t.mean()), "std_ms": float(t.std()), "min_ms": float(t.min()),
            "p95_ms": float(np.percentile(t, 95))}
