# JOSS paper outline — cclib-mojo

Working title: **cclib-mojo: Mojo-accelerated electron-density and
wavefunction-on-grid evaluation for cclib**

Target venue: [Journal of Open Source Software](https://joss.theoj.org/).
Scope of this document: section-by-section skeleton for `paper.md` (plus
`paper.bib`), following the
[JOSS paper format](https://joss.readthedocs.io/en/latest/paper.html)
(required sections: Summary, Statement of need, State of the field,
Software design, Research impact statement, AI usage disclosure,
Acknowledgements, References; 750–1750 words total).

Status: outline only. No submission has been made. Every number below is
already measured and published in
[`python/cclib_mojo/README.md`](../python/cclib_mojo/README.md) in this
repository — do not invent new ones; re-measure if the hardware or
software environment changes.

---

## 1. Metadata (YAML frontmatter skeleton)

```yaml
---
title: "cclib-mojo: Mojo-accelerated electron-density and wavefunction-on-grid evaluation for cclib"
tags:
  - Python
  - Mojo
  - computational chemistry
  - quantum chemistry
  - electron density
  - Gaussian basis functions
  - cube files
authors:
  # JOSS requires named individuals (given-names/surname), not
  # organizations. Fill in the actual author(s) of record with ORCID.
  - name: TBD (individual author of record, with ORCID)
    affiliation: "1"
affiliations:
  - name: Algenta
    index: 1
date: TBD  # e.g. 19 September 2026 (format: %e %B %Y)
bibliography: paper.bib
---
```

## 2. Summary (~100 words — non-specialist audience, minimal jargon)

Key points to hit:

- Computational-chemistry programs produce molecular orbitals and
  electron densities as numbers on a grid; chemists visualize and
  analyze them as 3-D fields (cube files).
- cclib is the standard open library for turning quantum-chemistry
  logfiles into such grids, but its grid evaluation runs an interpreted
  inner loop and is slow for production-size grids.
- cclib-mojo is a drop-in accelerator for that path: a compiled Mojo
  kernel behind a thin Python wrapper, with identical results and a
  pure-NumPy fallback on platforms without a prebuilt binary.
- One sentence of headline performance: 72× faster than the fastest
  backend cclib ships today (pyquante2) on a reference benzene grid,
  with agreement to 1e-12.

## 3. Statement of need (~150 words)

- Problem: `cclib.method.volume`'s `Volume.wavefunction()` /
  `electrondensity()` evaluates contracted Gaussian basis functions one
  grid point at a time through optional PyQuante backends; a single
  100³ orbital grid costs minutes (measured: 345.4 s on the PyQuante1
  amplitude path), which makes cube generation the bottleneck in
  visualization and analysis pipelines built on cclib.
- Audience: computational and theoretical chemists using cclib for
  orbital/density visualization, cube-file generation, and grid-based
  post-analysis; educators preparing visual material; developers of
  higher-throughput pipelines that embed cclib.
- Why existing options fall short: PyQuante1 cannot run on Python 3;
  pyquante2 vectorizes but still pays per-(basis function, grid)
  temporaries; neither offers a compiled separable-Gaussian kernel.
- Positioning: cclib-mojo keeps cclib's conventions exactly
  (`convertor` constants, `sym2powerlist` shell order, `Volume.data`
  layout) so results are drop-in equivalent, and it requires no new
  hard dependency (NumPy only) and no Mojo toolchain at install time.

## 4. State of the field (~120 words)

- cclib [@oboyle2008cclib] dispatches its volume path to optional
  PyQuante 1 / pyquante2 backends via `find_package` (`_check_pyquante`);
  pyquante2's NumPy-vectorized `cgbf.mesh` is the fastest backend cclib
  ships today.
- Adjacent tools that generate orbital/density grids outside cclib:
  QM-program-native utilities (Gaussian `cubegen`, ORCA `orca_plot`),
  standalone analyzers (Multiwfn), and PySCF's grid tools. These do not
  serve the cclib use case (parser-agnostic post-processing of any
  supported package's logfile).
- Build-vs-contribute justification: the contribution is a
  drop-in-compatible backend for an existing, cited community tool —
  the same optional-backend pattern cclib already uses — rather than a
  new ecosystem. Upstreaming a `mojo` backend to cclib has been
  proposed to the maintainers
  ([cclib/cclib#1909](https://github.com/cclib/cclib/issues/1909));
  the independent package remains usable regardless.

## 5. Software design (~200 words)

- **Batch-shaped C ABI**: one FFI call evaluates a whole 3-D grid for
  one MO (or accumulates the density over all occupied MOs); FFI
  overhead is per call, not per point.
- **Kernel**: precomputes the exactly separable axis factors
  `exp(-a·dx²)·exp(-a·dy²)·exp(-a·dz²)` once per call, then one SIMD
  fused multiply-add per (grid point, primitive) in compiled float64,
  grid row kept cache-resident. Single-threaded for determinism
  (multithreading is future work and would multiply the speedup).
- **Reference-matching arithmetic**: normalization constants computed
  in Python with PyQuante's exact formulas and operation order
  (Taketa–Huzinaga–O-ohata [@taketa1966] eqs. 2.2 and 2.12); the kernel
  evaluates the textbook contracted Cartesian Gaussian in IEEE-754
  float64 and skips exactly-zero MO coefficients, the same
  `abs(c) > 0.0` rule cclib uses.
- **Dual backend with handshake**: wrapper checks
  `gaussgridmojo_abi_version()` before evaluating; a mismatch or a
  missing shared library falls back cleanly to a vendored clean-room
  NumPy reference (`CCLIB_MOJO_DISABLE_NATIVE=1` forces it; the test
  suite runs both backends).
- **Packaging**: per-platform wheels (macOS arm64, Linux x86_64),
  self-contained — the Mojo runtime is vendored into the wheel and load
  paths rewritten with `delocate` (macOS) / `auditwheel repair`
  (Linux). Other platforms (e.g. Windows) run the NumPy fallback
  silently and correctly. Clean-room: cclib and PyQuante are
  test/benchmark references only, never runtime dependencies.

## 6. Benchmark and correctness methodology (~250 words, one table)

Present the measured table verbatim from the package README
(workload: benzene, 6-31G\*, 12 atoms, 102 contracted Cartesian basis
functions, 192 primitives, basis data from PyQuante 1.6.5's basis
library; seeded MO coefficients; median of 5 runs; single-threaded;
Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14, NumPy 2.5.3,
Mojo 1.1.0; 2026-09-19):

| workload | reference (s) | cclib-mojo (s) | speedup |
|---|---:|---:|---:|
| wavefunction, 1 MO, 50³ — cclib `Volume.wavefunction()` on pyquante2 backend | 0.689 | 0.0095 | 72× |
| wavefunction, 1 MO, 50³ — PyQuante1 amplitude path | 32.23 | 0.0126 | 2,567× |
| wavefunction, 1 MO, 100³ — PyQuante1 amplitude path | 345.37 | 0.0491 | 7,033× |
| electron density, 3 MOs, 50³ — PyQuante1 amplitude path | 63.59 | 0.0335 | 1,898× |
| wavefunction, 1 MO, 50³ — cclib-mojo NumPy fallback | 32.23 (PyQuante1) | 0.1691 | 191× |

Methodology notes to include:

- The PyQuante1 baseline is a verbatim Python-3 transcription of
  PyQuante1's `CGBF.amp` amplitude path (`tests/pyquante1_oracle.py` in
  the repo; PyQuante1 itself cannot execute on Python 3), driven exactly
  the way cclib's `pyamp()` drives it — same formulas, same operation
  order, same per-point interpreter cost profile. The pyquante2 baseline
  is cclib's real `Volume.wavefunction()`, unmodified.
- Correctness gate: max abs diff 1.5e-12 vs the PyQuante reference,
  asserted before every timing run; differential-suite tolerance 1e-10
  relative, measured agreement at the 1e-13–1e-16 level.
- End-to-end equivalence with cclib's own
  `Volume.wavefunction()`/`electrondensity()` on the pyquante2 backend
  (same grid points, normalization, cube-file ordering); max abs diff
  8.4e-13 on the 50³ workload.
- Per-call setup is comparable on both sides (PyQuante1 `getbfs`
  2.09 ms vs cclib-mojo flatten 1.48 ms), so the table reflects the
  evaluation loop, not setup asymmetry.
- Reproduce: `pixi run bench-cclib` (`benchmarks/bench_gaussgrid.py`).

## 7. Research impact statement (~100 words)

JOSS asks for realized impact or credible near-term significance —
use the concrete, verifiable signals (do not overclaim adoption):

- Measured, reproducible benchmark (above) with a correctness gate
  asserted before every timing run.
- 38 differential tests (19 tests run on both backends — native and
  forced NumPy fallback) covering s/p/d/f shells (STO-3G H₂O, 6-31G\*
  d on O, cc-pVTZ f on C), multiple atoms, MO amplitude vs summed
  density, anisotropic and origin-offset grids, exact-zero coefficient
  handling, grid-point ordering, and validation errors, plus the
  end-to-end cclib equivalence tests.
- CI workflow running build → differential tests (both backends) →
  wheel self-containment repair → hermetic install smoke
  (`.github/workflows/ci-cclib.yml`).
- Outreach: optional-backend proposal opened with the cclib
  maintainers (link the issue); package prepared for independent use
  either way.

## 8. Research use cases in quantum chemistry (~120 words)

Frame as "the things cclib's volume path is used for, now 1–3 orders of
magnitude cheaper":

- Orbital and electron-density cube files for visualization
  (VMD, PyMOL, ChimeraX, Jmol) — including fine grids (100³+) that are
  currently impractical from cclib.
- Grid-based post-analysis on `Volume.data`: numerical integration
  (`Volume.integrate()`), density differencing, electrostatic-potential
  evaluation pipelines that start from cclib-parsed data.
- Teaching: interactive exploration of orbital shape/size vs basis set
  in classroom settings, where minute-long evaluations break the flow.
- Pipelines: batch cube generation across many logfiles (cclib's core
  use case — parser-agnostic post-processing) for screening or
  dataset construction.

## 9. Availability (~60 words)

- Source: <https://github.com/thyn-ai/mojo-kernels>
  (`python/cclib_mojo`, kernel `kernels/gaussgrid`), Apache-2.0,
  © 2026 Algenta.
- Install: `pip install cclib-mojo` — per-platform self-contained
  wheels (macOS arm64, Linux x86_64); NumPy fallback everywhere else;
  NumPy is the only hard dependency; no Mojo toolchain required.
- Docs and quickstart: package README + `quickstart.py` (water,
  STO-3G); tests: `tests/test_gaussgrid_*.py`;
  benchmark: `benchmarks/bench_gaussgrid.py`.
- Archival: mint a Zenodo DOI for the release tag before submission
  (JOSS archives the paper itself; a software archive DOI is expected
  for the "Citing" path the README currently marks as forthcoming).

## 10. AI usage disclosure (required by JOSS)

Draft honestly before submission. The software and this documentation
were produced with generative-AI assistance (Algenta's agent-based
kernel program). State which parts were AI-generated and how
correctness was verified: all numerical claims come from a differential
test suite against the PyQuante/cclib references run on both backends
and gated in CI; benchmark numbers are machine-measured and
reproducible via `pixi run bench-cclib`.

## 11. Acknowledgements

- Algenta (development).
- State explicitly whether any grant funding applies (none currently —
  say "no external funding" if so).
- Optional thanks to the cclib and PyQuante authors for the reference
  implementations used as test oracles.

## 12. References (`paper.bib` skeleton)

```bibtex
@article{oboyle2008cclib,
  title   = {cclib: A library for package-independent computational chemistry algorithms},
  author  = {O'Boyle, Noel M. and Tenderholt, Adam L. and Langner, Karol M.},
  journal = {Journal of Computational Chemistry},
  year    = {2008},
  volume  = {29},
  number  = {5},
  pages   = {839--845},
  doi     = {10.1002/jcc.20823}
}

@article{taketa1966,
  title   = {Gaussian-Expansion Methods for Molecular Integrals},
  author  = {Taketa, Hiroshi and Huzinaga, Sigeru and O-ohata, Kiyosi},
  journal = {Journal of the Physical Society of Japan},
  year    = {1966},
  volume  = {21},
  number  = {11},
  pages   = {2313--2324},
  doi     = {10.1143/JPSJ.21.2313}
}

@misc{pyquante,
  title  = {PyQuante: Python Quantum Chemistry},
  author = {Muller, Richard P.},
  url    = {http://pyquante.sourceforge.net/},
  note   = {Version 1.6.5; basis library used for test fixtures}
}

@misc{pyquante2,
  title = {pyquante2: Python Quantum Chemistry, rewritten},
  author = {Muller, Richard P.},
  url    = {https://github.com/rpmuller/pyquante2}
}

@misc{mojo,
  title = {Mojo programming language},
  author = {{Modular}},
  url    = {https://www.modular.com/mojo},
  note   = {Version 1.1.0}
}

@article{numpy2020,
  title   = {Array programming with NumPy},
  author  = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, St{\'e}fan J. and others},
  journal = {Nature},
  year    = {2020},
  volume  = {585},
  pages   = {357--362},
  doi     = {10.1038/s41586-020-2649-2}
}
```

## 13. Submission-readiness checklist (JOSS review criteria)

- [x] OSI-approved license (Apache-2.0, in repo root `LICENSE`).
- [x] Statement of need visible in the package README.
- [x] Installation instructions + quickstart (`pip install`, `quickstart.py`).
- [x] Automated tests (38 differential tests, both backends) + CI.
- [x] Functionality documented (README "How it works", fallback semantics,
      scope and limitations).
- [ ] Named individual author(s) of record with ORCID (JOSS does not
      accept organizational authorship).
- [ ] `paper.md` + `paper.bib` written from this outline; word count
      750–1750; compile check via the Open Journals GitHub Action or the
      `openjournals/inara` Docker image.
- [ ] Zenodo archive DOI for the release (README "Citing" section then
      updated to reference it).
- [ ] Community guidelines (CONTRIBUTING) at repo or package level —
      currently absent; add before submission.
- [ ] Confirm redistribution terms for Modular's Mojo runtime binaries
      inside wheels (flagged in the package README) before the public
      PyPI release the paper will reference.
- [x] Link the cclib/cclib optional-backend issue from "State of the
      field" / impact statement — opened as
      [cclib/cclib#1909](https://github.com/cclib/cclib/issues/1909)
      (2026-09-19).
