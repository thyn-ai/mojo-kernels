#!/usr/bin/env python3
"""Arena-exact BM25 three-way duel + layer-by-layer autopsy (Xeon/Space edition).

Replicates the hf-speed-arena BM25 duel cell EXACTLY (arena/duels.py +
arena/corpora.py + arena/harness.py at arena harness v0.1.0):

  * same seeded corpus: ``bm25_corpus(SEED + n_docs, n_docs)`` — Zipf-ish
    20k-term vocab, 30-60 tokens/doc;
  * same queries: ``bm25_queries(SEED + n_docs + 10, terms, 10, 10)`` — 10
    queries x 10 terms over the corpus' unique terms (plus one unseen term in
    every 4th query);
  * same lanes:
      - rank_bm25 0.2.2:  np.stack([ref.get_scores(q) for q in queries])
      - bm25s 0.3.11:     np.stack([bs.get_scores(q) for q in queries])
                          (method='lucene', float32, compile()d numba scorer)
      - bm25-mojo:        np.asarray(ours.get_scores_batch(queries))
  * same protocol: gate call doubles as cold measurement; warm = median of 5
    interleaved round-robin reps (A/B/C, A/B/C, ...); spread = min/max;
  * same gate: max|diff| vs rank_bm25 <= 1e-8, asserted before timing;
  * numba scorer pre-warmed once at process start (the Space does this at
    boot; its one-time JIT cost is measured there, not in duel cells).

Modes:
  (default)   arena cells S/M/L (1k/5k/20k docs, 10 terms) + sweep
              (1k/5k/10k/20k docs x 5/10/20 terms), index-build times included
  --layers    staged per-layer timing of both warm paths (token->id mapping,
              array packing, score-buffer calloc, FFI/numba dispatch, kernel
              walk, np.stack), plus allocation/dispatch microcosts
  --cell M    run one arena size only (S/M/L)

Environment knobs: BM25_XRAY_REPS (warm reps, default 5),
BM25_XRAY_INDEX_REPS (index-build repetitions, default 1).

This script is self-contained on purpose: it runs identically on a dev
machine, in CI, and inside a Space-matched container. Do NOT import the
arena package from here; keep the protocol code verbatim-identical instead.
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import statistics
import time

import numpy as np

# ---------------------------------------------------------------------------
# Verbatim from arena/corpora.py (hf-speed-arena, harness v0.1.0) — do not edit
# ---------------------------------------------------------------------------

BASE_SEED = 20260919
BM25_VOCAB = 20_000


def bm25_corpus(seed: int, n_docs: int) -> list[list[str]]:
    rng = random.Random(seed)
    vocab = [f"w{i:05d}" for i in range(BM25_VOCAB)]
    weights = [1.0 / (i + 1) for i in range(BM25_VOCAB)]  # Zipf-ish
    return [rng.choices(vocab, weights, k=rng.randint(30, 60)) for _ in range(n_docs)]


def bm25_queries(seed: int, terms: list[str], q_len: int, n_queries: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [
        [rng.choice(terms) for _ in range(q_len)] + (["zzz_unseen"] if i % 4 == 0 else [])
        for i in range(n_queries)
    ]


# Arena duel constants (arena/duels.py)
BM25_N_QUERIES = 10
BM25_Q_LEN = 10
BM25_ATOL = 1e-8
ARENA_SIZES = {"S": 1_000, "M": 5_000, "L": 20_000}

WARM_REPS = int(os.environ.get("BM25_XRAY_REPS", "5"))
INDEX_REPS = int(os.environ.get("BM25_XRAY_INDEX_REPS", "1"))

SWEEP_DOCS = [1_000, 5_000, 10_000, 20_000]
SWEEP_QLENS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Hardware / environment disclosure (mirrors arena.harness.hardware_footer)
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _cgroup_cpus() -> str:
    """Effective CPU limit from cgroup v2 (or v1), e.g. '2.0 (cgroup v2 cpu.max)'."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            return f"{int(quota) / int(period):.2f} (cgroup v2 cpu.max)"
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as fh:
            period = int(fh.read().strip())
        if quota > 0:
            return f"{quota / period:.2f} (cgroup v1 cfs)"
    except (OSError, ValueError):
        pass
    return "unlimited/unknown"


def _cpu_flags() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].split()
                    isa = [f for f in ("avx512f", "avx512bw", "avx512vl", "avx2", "fma", "sse4_2") if f in flags]
                    return " ".join(isa)
    except OSError:
        pass
    return "n/a"


def env_report() -> str:
    import numba
    import numpy
    import bm25s

    lines = [
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- cpu: {_cpu_model()}",
        f"- isa flags: {_cpu_flags()}",
        f"- os.cpu_count: {os.cpu_count()} | cgroup-effective vCPU: {_cgroup_cpus()}",
        f"- python: {platform.python_version()} | numpy: {numpy.__version__} | "
        f"numba: {numba.__version__} | bm25s: {bm25s.__version__}",
        f"- platform: {platform.platform()}",
    ]
    try:
        import bm25_mojo

        info = bm25_mojo.backend_info()
        lines.append(
            f"- bm25_mojo: {bm25_mojo.__version__} native={info['native_available']} "
            f"({info.get('native_source') or info.get('error')})"
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- bm25_mojo: UNAVAILABLE ({exc})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workload + lanes (verbatim lane callables from arena/duels.py)
# ---------------------------------------------------------------------------


def build_workload(n_docs: int, q_len: int = BM25_Q_LEN):
    corpus = bm25_corpus(BASE_SEED + n_docs, n_docs)
    terms = list({t for doc in corpus for t in doc})
    # Arena seed formula for the 10-term cells is SEED + n_docs + BM25_Q_LEN;
    # sweep cells follow the same family (seed varies with q_len, as in
    # benchmarks/bench_bm25.py).
    queries = bm25_queries(BASE_SEED + n_docs + q_len, terms, q_len, BM25_N_QUERIES)
    return corpus, terms, queries


def make_indexes(corpus):
    """Build all three indexes; returns (ref, bs, ours, t_ref, t_bs, t_ours)."""
    import bm25s
    from rank_bm25 import BM25Okapi as RefBM25Okapi

    from bm25_mojo import BM25Okapi

    t0 = time.perf_counter()
    ref = RefBM25Okapi(corpus)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    bs = bm25s.BM25(dtype="float32")  # method='lucene' — the bm25s default
    bs.index(corpus, show_progress=False)
    bs.compile()  # numba scorer — the config bm25s recommends for speed
    t_bs = time.perf_counter() - t0

    t0 = time.perf_counter()
    ours = BM25Okapi(corpus)
    t_ours = time.perf_counter() - t0
    return ref, bs, ours, t_ref, t_bs, t_ours


def lane_fns(ref, bs, ours, queries):
    """The exact arena lane callables."""
    return {
        "rank_bm25": lambda: np.stack([ref.get_scores(q) for q in queries]),
        "bm25s": lambda: np.stack([bs.get_scores(q) for q in queries]),
        "bm25_mojo": lambda: np.asarray(ours.get_scores_batch(queries)),
    }


def run_cell(n_docs: int, q_len: int = BM25_Q_LEN, label: str = ""):
    """One duel cell under the arena protocol. Returns a result dict."""
    corpus, _terms, queries = build_workload(n_docs, q_len)
    ref, bs, ours, t_ref, t_bs, t_ours = make_indexes(corpus)
    lanes = lane_fns(ref, bs, ours, queries)

    # --- gate (doubles as cold) ---
    outputs, cold = {}, {}
    for name, fn in lanes.items():
        t0 = time.perf_counter()
        outputs[name] = fn()
        cold[name] = time.perf_counter() - t0
    worst = float(np.max(np.abs(outputs["bm25_mojo"] - outputs["rank_bm25"])))
    if worst > BM25_ATOL:
        raise SystemExit(f"GATE FAIL at {n_docs} docs/{q_len}t: max|diff| {worst:.3e} > {BM25_ATOL}")

    # --- warm: interleaved round-robin, median of WARM_REPS ---
    samples = {name: [] for name in lanes}
    for _rep in range(WARM_REPS):
        for name, fn in lanes.items():
            t0 = time.perf_counter()
            fn()
            samples[name].append(time.perf_counter() - t0)

    warm = {name: statistics.median(s) for name, s in samples.items()}
    warm_min = {name: min(s) for name, s in samples.items()}
    warm_max = {name: max(s) for name, s in samples.items()}
    best_inc = min(warm["rank_bm25"], warm["bm25s"])
    hero = best_inc / warm["bm25_mojo"]
    return {
        "n_docs": n_docs,
        "q_len": q_len,
        "label": label,
        "gate_maxdiff": worst,
        "index_s": (t_ref, t_bs, t_ours),
        "cold_s": cold,
        "warm_s": warm,
        "warm_min_s": warm_min,
        "warm_max_s": warm_max,
        "hero": hero,
        "incumbent": "bm25s" if warm["bm25s"] <= warm["rank_bm25"] else "rank_bm25",
    }


def print_cell(r):
    nq = BM25_N_QUERIES
    t_ref, t_bs, t_ours = r["index_s"]
    label = r["label"] or f"{r['n_docs']:,} docs · {nq}q × {r['q_len']}t"
    print(f"\n### {label} (gate max|diff| {r['gate_maxdiff']:.2e})")
    print(f"index build: rank_bm25 {t_ref:.3f}s · bm25s(+numba) {t_bs:.3f}s · bm25_mojo {t_ours:.3f}s")
    print(f"{'lane':<12} {'cold ms':>9} {'warm med ms':>12} {'min':>9} {'max':>9} {'ms/query':>9}")
    for name in ("rank_bm25", "bm25s", "bm25_mojo"):
        print(
            f"{name:<12} {1e3 * r['cold_s'][name]:>9.3f} {1e3 * r['warm_s'][name]:>12.3f} "
            f"{1e3 * r['warm_min_s'][name]:>9.3f} {1e3 * r['warm_max_s'][name]:>9.3f} "
            f"{1e3 * r['warm_s'][name] / nq:>9.4f}"
        )
    verdict = "WIN" if r["hero"] > 1.0 else "LOSS"
    print(f"hero: {r['hero']:.3f}× vs {r['incumbent']} (warm median of {WARM_REPS}) — {verdict}")


# ---------------------------------------------------------------------------
# --layers: staged per-layer timing of both warm paths
# ---------------------------------------------------------------------------


def layers(n_docs: int, q_len: int = BM25_Q_LEN, reps: int = 30):
    """Time each layer of the warm path separately, ours vs bm25s.

    The staged re-implementations below mirror bm25s 0.3.11's
    get_scores -> get_tokens_ids -> get_scores_from_ids -> njit scorer chain
    and bm25_mojo's get_scores_batch -> NativeIndex.score_batch chain step by
    step; each stage's output is asserted identical to the real lane's before
    any timing.
    """
    corpus, _terms, queries = build_workload(n_docs, q_len)
    ref, bs, ours, *_ = make_indexes(corpus)
    lanes = lane_fns(ref, bs, ours, queries)

    # ---------- staged bm25s path (mirrors bm25s/__init__.py 0.3.11) -------
    data = bs.scores["data"]
    indices = bs.scores["indices"]
    indptr = bs.scores["indptr"]
    num_docs = bs.scores["num_docs"]
    s_dtype = np.dtype(bs.dtype)
    i_dtype = np.dtype(bs.int_dtype)
    jit_fn = bs._compute_relevance_from_scores  # njit'd after compile()

    def bs_staged():
        outs = []
        for q in queries:
            qids = bs.get_tokens_ids(q)  # stage: vocab mapping
            arr = np.asarray(qids, dtype=i_dtype)  # stage: asarray
            _m = int(arr.max(initial=0))  # stage: max + bounds check
            if _m >= len(indptr) - 1:
                raise ValueError("token id out of range")
            sc = jit_fn(  # stage: numba call (zeros + fancy-index + walk inside)
                data=data, indptr=indptr, indices=indices, num_docs=num_docs,
                query_tokens_ids=arr, dtype=s_dtype,
            )
            if bs.nonoccurrence_array is not None:
                sc = sc + bs.nonoccurrence_array[arr].sum()
            outs.append(sc)
        return np.stack(outs)  # stage: stack

    real = lanes["bm25s"]()
    staged = bs_staged()
    assert np.array_equal(real, staged), "staged bm25s path diverges from real lane"
    has_nonocc = bs.nonoccurrence_array is not None

    def _asarray_max(qids):
        arr = np.asarray(qids, dtype=i_dtype)
        return arr, int(arr.max(initial=0))

    # ---------- staged bm25_mojo path (mirrors core.py + _native.py) -------
    # Two marshal conventions exist in the wild: the pre-0.1.4 wrapper bound
    # POINTER(c_int) argtypes and passed .ctypes.data_as(...) objects; the
    # 0.1.4+ wrapper declares c_void_p and passes integer .ctypes.data. The
    # staged call must match the INSTALLED wheel's convention.
    ni = ours._native_index
    vocab_get = ours._vocab.get
    lib = ni._lib
    batch_fn = ni._batch_fn
    handle = ni._handle
    n_docs_ni = ni._n_docs
    flat_abi = hasattr(ni, "score_batch_flat")

    import ctypes as _ct

    i32p, i64p, f64p = (
        _ct.POINTER(_ct.c_int32),
        _ct.POINTER(_ct.c_int64),
        _ct.POINTER(_ct.c_double),
    )

    def _flat_pack():
        flat = []
        offsets = [0]
        extend = flat.extend
        add_offset = offsets.append
        for q in queries:
            extend(v for v in map(vocab_get, q) if v is not None)
            add_offset(len(flat))
        return (
            np.array(flat, dtype=np.int32),
            np.array(offsets, dtype=np.int64),
        )

    def _old_pack():
        qids_list = [
            np.array([v for v in map(vocab_get, q) if v is not None], dtype=np.int32)
            for q in queries
        ]
        counts = np.fromiter((len(q) for q in qids_list), dtype=np.int64, count=len(qids_list))
        qids_flat = np.concatenate([np.ascontiguousarray(q, dtype=np.int32) for q in qids_list])
        offsets = np.zeros(len(qids_list) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return qids_flat, offsets

    def _batch_call(qids_flat, offsets, panel):
        if flat_abi:
            return batch_fn(handle, qids_flat.ctypes.data, offsets.ctypes.data,
                            len(queries), panel.ctypes.data)
        return batch_fn(handle, qids_flat.ctypes.data_as(i32p),
                        offsets.ctypes.data_as(i64p), len(queries),
                        panel.ctypes.data_as(f64p))

    def mojo_staged():
        qids_flat, offsets = (_flat_pack() if flat_abi else _old_pack())
        panel = np.zeros((len(queries), n_docs_ni), dtype=np.float64)  # stage: calloc
        rc = _batch_call(qids_flat, offsets, panel)  # stage: marshal + kernel walk
        assert rc == 0
        return np.asarray(panel)

    real_m = lanes["bm25_mojo"]()
    staged_m = mojo_staged()
    assert np.array_equal(real_m, staged_m), "staged mojo path diverges from real lane"

    def timeit(fn, reps=reps):
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    def time_parts(parts, reps=reps):
        """parts: list of (name, setup_fn->ctx, body_fn(ctx)). Time each body."""
        totals = {name: [] for name, _, _ in parts}
        for _ in range(reps):
            for name, setup, body in parts:
                ctx = setup()
                t0 = time.perf_counter()
                body(ctx)
                totals[name].append(time.perf_counter() - t0)
        return {name: statistics.median(s) for name, s in totals.items()}

    # bm25s stages (whole-lane steady state)
    bs_parts = [
        ("vocab map (10q)", lambda: None,
         lambda _c: [bs.get_tokens_ids(q) for q in queries]),
        ("asarray+max (10q)", lambda: [bs.get_tokens_ids(q) for q in queries],
         lambda ql: [_asarray_max(a) for a in ql]),
        ("numba jit call (10q)", lambda: [np.asarray(bs.get_tokens_ids(q), dtype=i_dtype) for q in queries],
         lambda arrs: [jit_fn(data=data, indptr=indptr, indices=indices, num_docs=num_docs,
                              query_tokens_ids=a, dtype=s_dtype) for a in arrs]),
        ("np.stack", lambda: [jit_fn(data=data, indptr=indptr, indices=indices, num_docs=num_docs,
                                     query_tokens_ids=np.asarray(bs.get_tokens_ids(q), dtype=i_dtype),
                                     dtype=s_dtype) for q in queries],
         lambda outs: np.stack(outs)),
    ]
    mojo_parts = [
        (("flat map+pack (10q)" if flat_abi else "qmap+arrays+pack (10q)"), lambda: None,
         lambda _c: (_flat_pack() if flat_abi else _old_pack())),
        ("panel np.zeros f64", lambda: None,
         lambda _c: np.zeros((len(queries), n_docs_ni), dtype=np.float64)),
        ("ffi call (walk inside)",
         lambda: (*(_flat_pack() if flat_abi else _old_pack()),
                  np.zeros((len(queries), n_docs_ni), dtype=np.float64)),
         lambda pk: _batch_call(pk[0], pk[1], pk[2])),
    ]

    # microcosts
    def micro():
        import ctypes

        f64p = ctypes.POINTER(ctypes.c_double)
        rows = []
        t = timeit(lambda: np.zeros((10, n_docs), dtype=np.float64), reps=50)
        rows.append((f"np.zeros (10x{n_docs}) f64 [{10 * n_docs * 8 / 1024:.0f} KiB]", t))
        t = timeit(lambda: [np.zeros(n_docs, dtype=np.float32) for _ in range(10)], reps=50)
        rows.append((f"10x np.zeros {n_docs} f32 [{10 * n_docs * 4 / 1024:.0f} KiB total]", t))
        arrs = [np.zeros(n_docs, dtype=np.float32) for _ in range(10)]
        t = timeit(lambda: np.stack(arrs), reps=50)
        rows.append(("np.stack 10x f32", t))
        arr = np.zeros((10, n_docs), dtype=np.float64)
        t = timeit(lambda: arr.ctypes.data_as(f64p), reps=200)
        rows.append((".ctypes.data_as (one)", t))
        t = timeit(lambda: arr.ctypes.data, reps=200)
        rows.append((".ctypes.data (one)", t))
        abi = lib.bm25mojo_abi_version
        t = timeit(lambda: abi(), reps=200)
        rows.append(("cached ctypes call (abi_version)", t))
        toks = [t for q in queries for t in q]
        t = timeit(lambda: [v for v in map(vocab_get, toks) if v is not None], reps=50)
        rows.append((f"dict.get map over {len(toks)} tokens", t))
        q0 = np.asarray(bs.get_tokens_ids(queries[0]), dtype=i_dtype)
        t = timeit(lambda: jit_fn(data=data, indptr=indptr, indices=indices, num_docs=num_docs,
                                  query_tokens_ids=q0, dtype=s_dtype), reps=50)
        rows.append(("bm25s numba call, 1 query (zeros+walk)", t))
        return rows

    print(f"\n## LAYERS @ {n_docs:,} docs · {len(queries)}q × {q_len}t  (median of {reps} staged reps)")
    print(f"bm25s nonoccurrence_array present: {has_nonocc}")
    whole_bs = timeit(lanes["bm25s"])
    whole_m = timeit(lanes["bm25_mojo"])
    print(f"whole lane: bm25s {1e3 * whole_bs:.3f} ms | bm25_mojo {1e3 * whole_m:.3f} ms "
          f"(ratio {whole_bs / whole_m:.3f}×)")

    print("\nbm25s stages (per 10-query lane call):")
    for name, med in time_parts(bs_parts).items():
        print(f"  {name:<28} {1e6 * med:>10.2f} us")
    print("bm25_mojo stages (per 10-query lane call):")
    for name, med in time_parts(mojo_parts).items():
        print(f"  {name:<28} {1e6 * med:>10.2f} us")
    print("microcosts:")
    for name, med in micro():
        print(f"  {name:<38} {1e6 * med:>10.2f} us")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def prewarm_numba():
    """The Space pre-warms the numba scorer once at boot; mirror that."""
    import bm25s

    tiny = bm25_corpus(BASE_SEED, 64)
    t0 = time.perf_counter()
    bm = bm25s.BM25(dtype="float32")
    bm.index(tiny, show_progress=False)
    bm.compile()
    bm.warmup_numba_scorer()
    bm.get_scores(["w00000"])
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", action="store_true", help="staged layer autopsy")
    ap.add_argument("--cell", choices=["S", "M", "L"], help="one arena size only")
    ap.add_argument("--sweep-only", action="store_true", help="skip arena S/M/L cells, run only the sweep")
    args = ap.parse_args()

    print("== environment ==")
    print(env_report())
    jit_s = prewarm_numba()
    print(f"- one-time numba JIT pre-warm (Space-boot equivalent): {jit_s:.2f}s")
    print(f"- warm reps per cell: {WARM_REPS} (interleaved A/B/C) · queries/cell: {BM25_N_QUERIES}")

    if args.layers:
        for n_docs in (ARENA_SIZES[args.cell],) if args.cell else (1_000, 5_000, 20_000):
            layers(n_docs)
        return

    results = []
    if args.cell:
        results.append(run_cell(ARENA_SIZES[args.cell], BM25_Q_LEN,
                                label=f"arena {args.cell}: {ARENA_SIZES[args.cell]:,} docs · 10q × 10t"))
    elif not args.sweep_only:
        for key, n_docs in ARENA_SIZES.items():
            results.append(run_cell(n_docs, BM25_Q_LEN,
                                    label=f"arena {key}: {n_docs:,} docs · 10q × 10t"))

    print("\n== sweep (same protocol; seed = SEED + n_docs + q_len) ==")
    sweep = []
    for n_docs in SWEEP_DOCS:
        for q_len in SWEEP_QLENS:
            r = run_cell(n_docs, q_len, label=f"{n_docs:,} docs · 10q × {q_len}t")
            sweep.append(r)
            print_cell(r)

    for r in results:
        print_cell(r)

    print("\n== summary table (warm, per batch of 10 queries) ==")
    print(f"{'cell':<26} {'rank_bm25 ms':>12} {'bm25s ms':>10} {'mojo ms':>9} "
          f"{'mojo vs bm25s':>14} {'verdict':>8}")
    for r in results + sweep:
        ratio = r["warm_s"]["bm25s"] / r["warm_s"]["bm25_mojo"]
        verdict = "WIN" if ratio > 1.0 else "LOSS"
        print(
            f"{r['label']:<26} {1e3 * r['warm_s']['rank_bm25']:>12.3f} "
            f"{1e3 * r['warm_s']['bm25s']:>10.3f} {1e3 * r['warm_s']['bm25_mojo']:>9.3f} "
            f"{ratio:>13.3f}× {verdict:>8}"
        )

    print("\n== index build summary (seconds) ==")
    print(f"{'cell':<26} {'rank_bm25':>10} {'bm25s+numba':>12} {'bm25_mojo':>10}")
    seen = set()
    for r in results + sweep:
        if r["n_docs"] in seen:
            continue
        seen.add(r["n_docs"])
        t_ref, t_bs, t_ours = r["index_s"]
        print(f"{r['n_docs']:>10,} docs {'':<12} {t_ref:>10.3f} {t_bs:>12.3f} {t_ours:>10.3f}")


if __name__ == "__main__":
    main()
