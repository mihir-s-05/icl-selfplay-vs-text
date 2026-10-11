# Research viewer

A static companion to the published self-play vs. text pretraining study. It follows the paper and the [research thread](https://x.com/rihim_s/status/2104760031573078463): a paper title, abstract, numbered figures, orange/blue comparisons, methods, sources, and a minimal footer.

## Local preview

From the repository root, with Python 3 and Node 18+:

```powershell
python viewer/tools/build.py
node --test viewer/tests/*.test.js
python -m http.server 4173 --bind 127.0.0.1 --directory viewer/dist
```

Open <http://127.0.0.1:4173/>. The same commands are available as `npm run build`, `npm test`, and `npm run preview` inside `viewer/`. Local building, testing, and preview need no dependency install. Keep the preview bound to loopback and serve only `viewer/dist`, never the repository root.

## What is included

| Artifact | Released rows | Viewer use |
| --- | ---: | --- |
| `headline.csv` | 1,191 | Model-size comparison, training progress, transfer |
| `tasks.csv` | 8,337 | Per-task comparison, task-specific training and transfer |
| `mcurves.csv` | 78,209 | Accuracy against number of examples |
| `valbpb.csv` | 1,248 | Individual DCLM and warm-start validation loss |
| `sp_bpb.csv` | 198 | Original paper's SP zero-shot DCLM loss |

The five original figures are included as SVG and PDF. Original CSVs, normalized JSONs, the source writeup, and a checksum manifest are downloadable. The JSON columns use dictionary indices for categorical fields; see `src/data.js:decodeTable` for decoding. Tokens retain the CSV unit: **billions of byte-level learner tokens**. Warm-start tokens count only the additional DCLM phase.

The build reads only an explicit allowlist of Git blobs at public commit `74e8f7be101e050c6982f2838f2d3ea980851f5f`. The commit and all five CSV Git blobs were checked against public GitHub during implementation. Git may convert line endings in the Windows working tree; the build copies the committed bytes. It does not read ignored results, local social figure drafts, private notes, `.env`, training state, or checkpoints. No training or evaluation modules are imported. No network requests are made by the build or by the viewer for its data.

The local released result download contains 331 evaluation JSON files. They remain external source links because the five tidy tables already provide the plotted measurements. Trial counts, task generation metadata, probabilities, and training configurations require the linked [Hugging Face result repository](https://huggingface.co/datasets/rihim/icl-selfplay-vs-text-results). Checkpoints are linked separately and are never loaded.

## Reading the comparisons

- Figure 1 compares SP and text at the five published matched token counts, with exact non-embedding parameter counts on a logarithmic axis. The nominal “100k” model has 65,728 non-embedding parameters.
- SP and UP bands/whiskers use a 95% Student t interval over the available seeds. Bounds are clipped only for drawing on a 0–1 accuracy axis; tables/downloads retain unclipped values. Single seeds get no interval. No DCLM seed interval or difference interval is inferred from 1–2 runs.
- Text mean markers and hollow individual text run markers show limited replication. Exact seed IDs can be selected; seeds are not paired across arms.
- UP is at the corresponding paper round, with 1,024 or 2,048 programs per round versus SP's 1,536. Its tokens are **not** equal to SP/text tokens. There is no released 24M UP checkpoint. The 100k UP tables have only two seeds at rounds 256 and 512; other plotted UP checkpoints have four. Intervals use the actual available seed count. Missing data is shown as missing, never as zero.
- Word headlines average classify-2, classify-4 and lookup-4. Word reversal remains in the per-task view. Zero-scoring addition, reversal and cipher tasks are preserved.
- Raw reverse uses per-byte accuracy. Other cells are greedy exact match; printable/word multi-byte answers require every byte to be correct. This distinction is present in axes and captions.
- Example curves use `log2(m + 1)`, so measured zero-example results remain visible. Missing example counts are not invented. Lines connect measurements without interpolating additional data rows.
- Transfer curves start at the first measured warm point, 0.05B added text tokens, after SP round 8191 (51.54B prior learner tokens). They do not draw the four-seed SP mean as if it were the single warm run's initial observation. Warm starts stop at 0.5B. Scratch means/hollow runs use all released text seeds unless filtered. This differs from original Fig. 4, which displays scratch seed-0 only.
- Loss curves show individual validation runs and the paper's zero-shot SP mean as distinct sources, in bits per byte. Lower is better. They are not ICL accuracy curves.

The explorer saves its view, model size, suite, task, seed, checkpoint and UP choice in URL parameters. That permits copying a specific view's URL. Figure 1's seed/UP visibility is a display setting and is not saved in the URL.

## Scope and material gaps

All conclusions are limited to this architecture, six nominal model sizes through 24M, narrow task formats, 4 SP seeds, 1–2 text seeds, and 1 warm-start seed. Learner tokens are matched; total compute, including the SP generator/RL update, is not. Warm-start and scratch learning rates differ. The viewer does not extrapolate scaling laws, calculate a new token speed-up, or infer a compute-matched result.

The compact tables do not retain trial counts, full prompt-level outputs, or candidate-answer probabilities. No prompt-level confidence interval or restricted-choice rescore can be recovered from them. SP text-loss data contains an aggregate rather than individual seed IDs. The source writeup is the authority for robustness checks and any speed-up interpretation. It has more qualified 3M catch-up ranges than the thread's compact “6–13× at 1M–3M” summary; the viewer avoids repeating that summary as a new measured statistic.

## Hosting fit

`wrangler.jsonc` configures a separate Cloudflare static-assets Worker at **https://icl.mihir-s.com/**. The portfolio links to that subdomain; it does not host a copy of the viewer. Cloudflare manages the subdomain's DNS and certificate through its [Custom Domain](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/) configuration.

From `viewer/`, with the pinned source commit available to Git:

```powershell
npm ci
npm test
npm run deploy:check
npm run deploy
```

Both deployment commands rebuild the allowlisted `dist/` output first. `deploy:check` checks packaging without publishing; `deploy` publishes only that directory. Wrangler needs a valid Cloudflare login or a suitably scoped `CLOUDFLARE_API_TOKEN`. Do not deploy the checkout or training directories.

No database, live model, GPU, secret, paid inference, or server API is needed. CSV/JSON downloads plus the source manifest provide useful read-only data access; adding a separate API would duplicate them. Large example data is loaded only when its view is selected. `_headers` provides a same-origin CSP, content-type protection, and a referrer policy. The Worker has no portfolio bindings or shared backend.

## Validation

Build invariants reject incorrect row counts, duplicate measurement keys and out-of-range accuracy. `tests/data.test.js` checks all normalized cells against released CSVs, checksums, known figure gaps, Student t intervals, zero tasks, exact seed coverage, UP round/token mapping, transfer progress, and the output allowlist. Browser checks cover the page identity, rendered figures, tab/selector changes, URL restoration, downloads, missing-data states, and desktop/phone layouts. Wide detailed figures and tables scroll within their own region on phones; overview figures stack vertically.

Source implementation lives in `src/`; reproducible output is ignored in `dist/`. The training/evaluation pipeline and existing uncommitted social figures are untouched.
