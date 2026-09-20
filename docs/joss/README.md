# JOSS submission — cclib-mojo

This directory holds the [Journal of Open Source
Software](https://joss.theoj.org/) (JOSS) submission draft for
`cclib-mojo` (`python/cclib_mojo` in this repository):

- `paper.md` — the paper, in the JOSS Markdown format
  ([paper format docs](https://joss.readthedocs.io/en/latest/paper.html)).
- `paper.bib` — the bibliography (BibTeX; all six entries are cited in
  the paper).

**Status: draft. No submission has been made.** Every performance and
correctness number in the paper is copied from the measured, published
values in [`python/cclib_mojo/README.md`](../../python/cclib_mojo/README.md)
(workload: benzene 6-31G\*, Apple M4 Max, 2026-09-19). Do not edit
numbers without re-measuring via `pixi run bench-cclib` and updating the
package README in the same commit.

## Blockers before submitting

These must be resolved first — the editor will bounce the submission
otherwise:

1. **Author of record.** JOSS does not accept organizational authorship.
   Replace `AUTHOR OF RECORD TBD` in the `paper.md` frontmatter with a
   named individual who made major contributions, with their ORCID
   (`0000-0000-0000-0000` is a placeholder). Affiliation stays
   `Algenta`.
2. **Archive DOI.** Mint a Zenodo DOI for the release tag being
   submitted, and update the "Citing" section of
   `python/cclib_mojo/README.md` (currently marked "forthcoming") to
   reference it. JOSS expects a software archive in addition to the
   paper's own DOI.
3. **Mojo runtime redistribution.** The wheels vendor Modular's runtime
   binaries (via `delocate` / `auditwheel repair`). The package README
   already flags that redistribution terms should be confirmed with
   Modular before the public PyPI release the paper references. Resolve
   before release.
4. **Re-verify the paper.** Word count must be 750–1750 (currently
   ~1,440 measured with the command below; check after any edit) and
   the PDF must compile (next section) — the draft compiles cleanly
   with `openjournals/inara` as of 2026-09-19.

Also decide *what* is being submitted: reviewers will install whatever
`pip install cclib-mojo` resolves to, so the PyPI release (or a clearly
documented test-PyPI/source install path) must exist by review time.

## Checking that the paper compiles

Word count (body only, excluding YAML frontmatter and the references
section):

```sh
awk '/^---$/{n++; next} n<2{next} /^# References/{exit} {print}' \
  docs/joss/paper.md | wc -w
```

Compile to PDF, either with Docker (the same `inara` image JOSS uses —
writes `paper.pdf` next to `paper.md`):

```sh
docker run --rm \
  --volume "$PWD/docs/joss:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

or with the [Open Journals GitHub
Action](https://github.com/marketplace/actions/open-journals-pdf-generator),
which uploads the PDF as a build artifact on each push.

## Submission walkthrough

1. Sign in at <https://joss.theoj.org/> with the submitting author's
   ORCID (this is why the author of record is blocker #1).
2. **Submit a paper**: provide the repository URL
   (`https://github.com/thyn-ai/mojo-kernels`), the software version /
   release tag, the paper location (`docs/joss/paper.md`), a suggested
   handling editor, and optional reviewer suggestions (check JOSS's
   conflict-of-interest policy first — no recent coauthors or same
   organization).
3. A **pre-review** issue opens on `openjournals/joss-reviews`. The
   track editor performs a scope check and assigns an editor, who
   recruits two or more reviewers.
4. The **review** issue opens. Review is public and checklist-driven;
   expect to respond to comments in the issue and to tag patch releases
   if reviewers find install or correctness problems. When all
   checkboxes are ticked, the editor recommends acceptance, the paper is
   compiled to PDF/JATS, Crossref-registered, and archived.

### What the reviewers will ask (the official checklist)

From the [JOSS review
checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)
and [review
criteria](https://joss.readthedocs.io/en/latest/review_criteria.html),
with where this repository stands today:

- **General**: source available at the repository URL ✓; plain-text
  OSI license (Apache-2.0, root `LICENSE`) ✓; submitting author made
  major contributions and the author list is appropriate — depends on
  blocker #1; clear research impact / scholarly significance — argued
  in the paper's Research impact statement.
- **Development history**: reviewers look for sustained development,
  open development over time (ideally ~6 months of public history with
  releases and external engagement), and community contributions. This
  repository is young — expect questions here; the honest answer is the
  public CI history, the measured benchmark, and the cclib upstreaming
  issue ([cclib/cclib#1909](https://github.com/cclib/cclib/issues/1909)).
- **Functionality**: installation proceeds as documented; functional
  claims confirmed; **performance claims confirmed** — reviewers may
  re-run `pip install cclib-mojo`, `quickstart.py`, the differential
  tests, and the benchmark on their own hardware. Numbers differ across
  machines; the README's reproduction path (`pixi run bench-cclib`) is
  the defense.
- **Documentation**: statement of need ✓ (package README); installation
  instructions with an automated dependency story ✓ (`pip install`,
  NumPy the only hard dependency); example usage ✓ (README quickstart +
  `quickstart.py`); functionality documented ✓ (README "How it works",
  fallback semantics, scope and limitations); automated tests ✓ (38
  differential test runs over both backends, `.github/workflows/ci-cclib.yml`);
  community guidelines ✓ (root `CONTRIBUTING.md`).
- **Paper**: required sections present (Summary, Statement of need,
  State of the field, Software design, Research impact statement, AI
  usage disclosure) ✓; writing quality; references complete with full
  venue names and DOIs where they exist ✓ (`paper.bib`).

## References

- JOSS paper format: <https://joss.readthedocs.io/en/latest/paper.html>
- JOSS review checklist:
  <https://joss.readthedocs.io/en/latest/review_checklist.html>
- JOSS review criteria:
  <https://joss.readthedocs.io/en/latest/review_criteria.html>
- JOSS submission site: <https://joss.theoj.org/>
- About JOSS itself: A. M. Smith, K. E. Niemeyer, D. S. Katz, et al.,
  "Journal of Open Source Software (JOSS): design and first-year
  review", *PeerJ Computer Science* 4:e147 (2018),
  <https://doi.org/10.7717/peerj-cs.147>
