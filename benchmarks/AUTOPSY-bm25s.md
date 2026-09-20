# Autopsy: bm25_mojo (v1 kernel) vs bm25s 0.3.11 (numba), layer by layer

Date: 2026-09-20. Machine: Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14,
numpy 2.5.3, numba 0.67.0, Mojo 1.1.0. Corpora: the shared seeded generators
from `benchmarks/bench_bm25.py` (Zipf-ish 20k-term vocab, 30-60 tokens/doc).
Measurement: `benchmarks/autopsy_bm25s_layers.py` + allocation probes run in
the harness's exact pattern (batches of 20 calls, each result dropped
immediately, median of 7 batches). The latter point matters: holding results
alive forces fresh pages and inflates allocation costs ~5-10x.

## 1. What bm25s actually does (from its installed source)

Index time (`bm25s/__init__.py:380-434`, `scoring.py`):

1. builds `idf_array` (float32) from document frequencies;
2. **`_build_scores_and_indices_for_matrix` computes the full BM25 score
   `idf * tf-component` for every posting, once, at index time**;
3. stores them as a **CSC score matrix** `scores = {data, indices, indptr}`:
   per term, a postings list of `(doc_id: int32, precomputed_score: float32)`
   — 8 bytes/posting.

Query time (`scoring.py:_compute_relevance_from_scores_jit_ready`, compiled
by numba after `bm.compile()`):

```python
scores = np.zeros(num_docs, dtype=float32)          # calloc, arena-recycled
for i in range(len(query_tokens_ids)):
    start, end = indptr[qid[i]], indptr[qid[i] + 1]
    for j in range(start, end):
        scores[indices[j]] += data[j]               # load f32, scatter-add
```

That is the entire hot path: **no division, no idf lookup, no doc_len gather
at query time** — everything multiplicative was baked at index time. It is
**single-threaded** (numba `njit`, no `prange`/`parallel` anywhere in the
package; verified in the installed source). The numpy variant does
`np.add.at` over concatenated postings (slow); the numba variant is the
recommended config and the one benchmarked.

Our v1 query path, by contrast, per query:

1. wrapper maps tokens to ids (Python dict), allocates `np.empty(f64)`;
2. kernel **eagerly memsets the whole f64 score buffer**;
3. per term: idf load, then per posting: gather `doc_len` (f64), evaluate the
   full BM25 formula **including one f64 division per posting** (SIMD x2
   lanes on NEON), scatter-add f64. 12 bytes/posting (i32 doc + f64 freq)
   plus the gathered doc_len reads.

## 2. Corpus shape (drives everything)

| docs | terms | postings (nnz) | avg df | postings/query (5 terms) | postings/query (20 terms) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 8,816 | 36,934 | 4.2 | 12 | 92 |
| 10,000 | 19,375 | 375,475 | 19.4 | 44 | 236 |
| 100,000 | 20,000 | 3,756,973 | 187.8 | 666 | 4,361 |

The walk is tiny at every cell; **fixed per-query costs dominate**, and they
are O(n_docs), not O(postings): the score-buffer initialization.

## 3. Measured layer costs (bench-harness pattern, immediate-free)

100k docs, 20-term queries, per query:

| layer | bm25_mojo v1 | bm25s (numba f32) |
|---|---:|---:|
| (a) token -> id mapping | ~1 us | ~0.7 us |
| (f) allocation + FFI/dispatch | np.empty 0.2 us + ctypes ~0.5 us | np.zeros + numba dispatch |
| (c) score-buffer init | **kernel memset 800 KB ~6.7 us** (incl. alloc+FFI) | **np.zeros 400 KB ~2.5 us** |
| (d) scoring walk (4,361 postings) | **~9.5-13.8 us** (division + doc_len gather, 12 B/posting, f64 out) | **~6-8.4 us** (pure gather-add, 8 B/posting, f32 out) |
| **total (measured)** | **17.1-23.1 us** | **11.8-13.0 us** |

Allocation micro-costs, immediate-free pattern (per call):

| alloc | 1k | 10k | 100k |
|---|---:|---:|---:|
| `np.empty` f64 | 0.37 us | 0.47 us | 0.17-6.0 us (arena-dependent) |
| `np.zeros` f64 | 0.42 us | ~4.0 us | **4.5 us (recycled arena)** |
| `np.zeros` f32 | 0.35 us | ~2.3 us | **2.5 us (recycled arena)** |

`np.zeros` on this numpy/macOS build is **not** page-lazy in steady state:
with immediate freeing the arena is recycled and the cost is a resident-page
memset, linear in bytes (~0.55 us/100KB). With results held alive it
degenerates to mmap + page faults (~25-50 us at 100k). Either way the f64
buffer costs 2x the f32 one — bm25s's structural edge at 100k docs.

## 4. Verdict per layer

- **(a) mapping**: both trivial; ours slightly heavier (list comp + np.array).
- **(b) idf lookup**: noise either way (L1-resident); bm25s avoids it via baking.
- **(c) zeroing**: our v1 loses ~2-3x (eager kernel memset f64 vs calloc-recycled f32).
  Fix: wrapper-side `np.zeros` (identical bytes, identical result), delete the
  kernel memset. Residual f64-vs-f32 byte gap (2x) is structural for a
  parity-exact f64 output array; it is ~2 us at 100k in harness conditions.
- **(d) walk**: our v1 loses ~1.7x per posting (f64 division + doc_len gather
  + 12 B/posting vs their 8 B/posting pure add). Fix: bake
  `idf * tf-component` into the CSR values at index build — the bm25s trick,
  done honestly in float64 with identical op order per posting (bit-exact
  parity by construction). Query-time walk becomes the same load+scatter-add
  as theirs, at 12 B/posting vs their 8.
- **(e) bandwidth vs compute**: v1 is compute-bound (division); v2 is
  bandwidth/L2-bound like bm25s. At 100k the f64 out array (800KB) lives in
  L2; f32 (400KB) stays hotter. Structural 1.5x bytes/posting against us.
- **(f) FFI/alloc**: ~1-2 us/query on both sides; batching amortizes it.
- **(g) index/query split**: bm25s pays the full formula once per posting at
  index time (its index build at 100k: 1.92s vs rank_bm25 0.42s). Baking
  moves our per-posting formula from per-query to per-index — the same
  honest trade. Our CSR construction additionally moves from Python into the
  kernel, cutting the v1 build overhead (1.50s at 100k) sharply.

## 5. Optimization plan (with expected layer impact)

- **(a) baked idf-weighted score matrix**: kills (b) and the division in (d);
  walk -> bm25s-equivalent gather-add.
- **(d) wrapper-side calloc init**: (c) 6.7 -> ~5 us at 100k, ~1-2 us at 10k.
- **(c) query batching** (`get_scores_batch`): one FFI call + one panel mmap
  + one mapping pass per 20 queries; kills (f) and (a) amortizable share.
- **(d2) generation stamps**: analyzed and **rejected** — `get_scores` must
  return a caller-owned dense array with true zeros in untouched docs; any
  stamp scheme either returns garbage (np.empty) or pays the same
  fault/memset cost when materializing zeros. The honest floor for (c) is
  bm25s's own floor: calloc + write touched docs only. (Raw-mmap lazy pages
  also lose: ~50 page faults at 16KB pages ~ 5-15 us > recycled calloc.)
- **(e) deterministic threading**: held in reserve for the 100k cells where
  the residual f64 byte gap could still leave us <1.0x; opt-in only, fixed-
  order doc-partitioned merge, bit-identical results. Deployed only if the
  measured grid still shows a losing cell after (a)+(c)+(d).

Results after each step are in the commit history and the README tables.

---

# Results after optimization (2026-09-20, same machine/harness)

## Final 9-cell grid vs bm25s (best-of-single/batch vs best-of-bm25s-variants)

| corpus | query terms | bm25s (numba) ms/query | bm25_mojo ms/query | speedup | winning path |
|---:|---:|---:|---:|---:|---|
| 1,000 | 5 | 0.0023 | 0.0014 | **1.71x** | batch |
| 1,000 | 10 | 0.0026 | 0.0016 | **1.61x** | batch |
| 1,000 | 20 | 0.0031 | 0.0020 | **1.60x** | batch |
| 10,000 | 5 | 0.0032 | 0.0017 | **1.95x** | batch |
| 10,000 | 10 | 0.0035 | 0.0019 | **1.84x** | batch |
| 10,000 | 20 | 0.0048 | 0.0029 | **1.66x** | batch |
| 100,000 | 5 | 0.0079 | 0.0055 | **1.43x** | batch |
| 100,000 | 10 | 0.0090 | 0.0068 | **1.33x** | batch |
| 100,000 | 20 | 0.0115 | 0.0105 | **1.10x** | batch |

Baseline before this work (v1 kernel, same harness): 0.56x-0.82x across the
grid. Parity gate after every step: max element-wise diff vs rank_bm25
3.6e-15 at 100k docs (gate: 1e-8); differential suite 74x2 green.
The 100k/20t cell was re-measured with 11 interleaved samples for stability:
median ratio 1.12x.

## Per-optimization win attribution (measured)

| step | what changed | measured effect |
|---|---|---|
| (a) baked idf-weighted CSR | idf*tf-component baked per posting at index time (f64, reference op order); query walk loses the division + doc_len gather | walk at 100k/20t: ~9.5-13.8 us -> ~7 us; v1->v2 single-query total at 100k/20t: 23.1 -> 14.2 us (with (d)) |
| (b) float32 storage | **measured and rejected**: f32-rounded weights with f64 accumulation give max abs error **9.4e-7** vs rank_bm25 at 100k docs — **94x over the 1e-8 gate**. f32 storage is incompatible with the parity contract; float64 kept. The 2x output-bytes tax vs bm25s-f32 is structural and was beaten on fixed costs instead |
| (c) query batching | `get_scores_batch` (one FFI call per panel); chunk rule: 10-query panels while panel <= 1 MB, whole batch otherwise | 10k/20t: 4.03 -> 2.9 us/query; 1k/20t: 2.8 -> 2.0; 100k/5t: 9.2 -> 5.5. Batch rows bit-identical to single-query calls (suite-asserted) |
| (d) no re-zeroing | kernel memset deleted; wrapper hands a numpy calloc buffer; per-query cost O(postings) not O(n_docs) | zeroing+FFI at 100k: 6.7 -> ~5 us; at 10k: kernel-side 6.99 us -> calloc 0.68 us. Generation stamps analyzed and rejected (see section 5) |
| (e) threading | **not deployed**: after (a)+(c)+(d)+(f) every cell is >= 1.10x single-threaded, so the opt-in complexity buys nothing the mandate requires. Reserve: if a future cell regresses below 1.0x, deterministic doc-partitioned partials merged in fixed order (bit-identical by construction) |
| (f) Python/FFI slimming | c_void_p args + `.ctypes.data`, cached bound methods, C-level map/filter | 10k/20t single: 5.44 -> 4.03 us; 10k/5t: 3.65 -> 2.83 us |
| build | CSR construction moved into the kernel (doc-major streams assembled with C-level iteration) | index build at 100k: 1.50 s -> 0.81 s (rank_bm25: 0.47 s, bm25s: 1.58 s) |

## The malloc panel-size effect (drives the batch chunk rule)

Measured on this machine (immediate-free tight loops, median of 9-11):
`np.zeros` cost is arena-recycled-memset up to roughly 1 MB (f64 800 KB:
4.5 us; 320-800 KB panels recycle well), while a 1.6 MB panel costs ~84 us
per call (fresh mapping + page faults) and 4-8 MB panels are worse than one
16 MB panel. The harness therefore chunks batches into <=1 MB panels when
that fits (10 rows x 80 KB at 10k docs), and uses one full-batch panel at
100k docs. Both patterns are public-API usage; the batch API takes any
query list.
