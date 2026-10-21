# CCF AIOps 2022 Detectability/Recovery Analysis — Methods and Provenance

This document describes the methodology behind the fault-detectability and recovery-time
analysis run on the CCF AIOps Challenge 2022 dataset (`data/raw/aiops2022/`), covering both
the training and test splits, at both pod/service level and node level. It is written to be
liftable into a paper's Methods section, and precise enough that the reported numbers can be
reproduced exactly from the scripts referenced below.

## 1. Dataset scope

### Training split

300 fault-injection events, drawn from 7 groundtruth files:

| File | Cloudbed | Date | Events |
|---|---|---|---|
| `groundtruth-k8s-1-2022-03-20.csv` | 1 | 2022-03-20 | 37 |
| `groundtruth-k8s-1-2022-03-21.csv` | 1 | 2022-03-21 | 43 |
| `groundtruth-k8s-2-2022-03-20.csv` | 2 | 2022-03-20 | 33 |
| `groundtruth-k8s-2-2022-03-21.csv` | 2 | 2022-03-21 | 55 |
| `groundtruth-k8s-3-2022-03-20.csv` | 3 | 2022-03-20 | 31 |
| `groundtruth-k8s-3-2022-03-21.csv` | 3 | 2022-03-21 | 51 |
| `groundtruth-k8s-3-2022-03-24.csv` | 3 | 2022-03-24 | 50 |
| **Total** | | | **300** |

Coverage is ragged by design, not by data loss: cloudbed1 and cloudbed2 each have fault
injections recorded only on 2022-03-20 and 2022-03-21; cloudbed3 additionally has
2022-03-24. Each row is a discrete fault-injection event with a single onset timestamp
(`timestamp`), a `level` (`pod`, `service`, or `node`), the `cmdb_id` of the affected
component, and a `failure_type` (raw Chinese label — see Section 2).

### Test split

241 fault-injection events, drawn from 5 groundtruth JSON files under
`初赛评分数据/groundtruth/` inside `test_data.7z`:

| Date folder | Events |
|---|---|
| 2022-05-01 | 45 |
| 2022-05-03 | 50 |
| 2022-05-05 | 46 |
| 2022-05-07 | 50 |
| 2022-05-09 | 50 |
| **Total** | **241** |

Two properties of the test split were verified before treating it as a holdout:

- **Zero date overlap with training.** The node-metric filename inside each test date
  folder embeds a misleading `MMDD` suffix (e.g. `2022-05-01/.../kpi_cloudbed1_metric_0324.csv`
  looks like March 24 data); the *actual* timestamps inside those files were checked directly
  and fall in April 30 – May 9, 2022 — entirely outside the training window
  (2022-03-19 through 2022-03-24). The filename suffix is a stale artifact from the file-naming
  script and should not be used to infer the date.
- **Identical service topology.** The set of distinct `service` values in
  `metric_service.csv` was compared between `2022-03-20-cloudbed1` (training) and
  `2022-05-01/cloudbed` (test): both contain exactly the same 11 entries
  (`adservice-grpc`, `adservice-http`, `cartservice-grpc`, `checkoutservice-grpc`,
  `currencyservice-grpc`, `emailservice-grpc`, `frontend-http`, `paymentservice-grpc`,
  `productcatalogservice-grpc`, `recommendationservice-grpc`, `shippingservice-grpc`), with
  zero services unique to either side.

Together these establish the test split as a **genuine temporal holdout on the same
system** (same microservice topology, later calendar time) — not a cross-deployment
generalization test. Any claim about "generalization" drawn from this train/test split
should be scoped to temporal generalization only.

## 2. Signal sources

### Pod/service-level events

Signal: `metric/service/metric_service.csv`, schema `service, timestamp, rr, sr, mrt, count`
(request rate %, success rate %, mean response time ms, request count), at 60-second
granularity. Streamed directly from the archives (see Section 5) — never fully extracted.

**cmdb_id → service resolution.** Groundtruth `cmdb_id` values do not always match
`metric_service.csv`'s `service` column directly:

- `level == "service"`: `cmdb_id` is already the base service name (e.g. `frontend`,
  `cartservice`) and is matched against every `service` entry whose name, with its
  `-grpc`/`-http` suffix stripped, equals `cmdb_id`.
- `level == "pod"`: `cmdb_id` has a trailing pod-instance suffix (e.g. `currencyservice-2`),
  stripped via regex `-\d+$` to obtain a base name candidate.
- **Shadow-replica pattern**: a subset of pod-level `cmdb_id`s follow
  `{service}2-{instance}` (e.g. `adservice2-0`, `shippingservice2-0`,
  `recommendationservice2-0`, `frontend2-0`, `cartservice2-0`, `checkoutservice2-0`,
  `productcatalogservice2-0`) — a second-replica naming convention that has no matching
  entry in `metric_service.csv` under its literal name. These are resolved by stripping the
  instance suffix, then, if the remainder ends in `2` and the string with that `2` removed is
  a known base service name, using the stripped name (`adservice2` → `adservice`). This
  pattern accounted for all 19 previously-unmapped pod events in the training split (see
  Section 5) and required no special-casing for the test split — the same function resolved
  its shadow-replica `cmdb_id`s (e.g. `checkoutservice2-0`) without modification.
- `level == "node"`: no resolution is attempted here; node-level events use the node signal
  source below instead, because no pod-to-node placement topology exists in this dataset to
  trace a node fault to a specific service.

### Node-level events

Signal: `metric/node` (one CSV per cloudbed/day for training —
`kpi_cloudbed{N}_metric_{MMDD}.csv`; one CSV per date for test, with the caveat above that
the `MMDD` in the test filename does not reflect the actual date), schema
`timestamp, cmdb_id, kpi_name, value`, `cmdb_id` ∈ `{node-1, ..., node-6}` — a direct match
against the groundtruth `cmdb_id`, no name resolution needed.

Each node-level `failure_type` is checked against a fixed set of relevant KPIs (all
confirmed present in the actual `kpi_name` column):

| failure_type (English) | `kpi_name`(s) checked |
|---|---|
| `node_cpu_fault` | `system.cpu.pct_usage`, `system.load.1` |
| `node_cpu_climb` | `system.cpu.pct_usage`, `system.load.1` |
| `node_disk_read_io_consumption` | `system.io.r_s`, `system.io.rkb_s`, `system.io.r_await` |
| `node_disk_write_io_consumption` | `system.io.w_s`, `system.io.w_await` |
| `node_disk_space_consumption` | `system.disk.used`, `system.disk.pct_usage` |
| `node_memory_consumption` | `system.mem.used`, `system.mem.pct_usage` |

An event is scored against every listed KPI independently (Section 3); `detected_any` is
true if any one of them detects, and `matched_kpis` in the per-event CSV records exactly
which KPIs had usable baseline data. This is a targeted, fault-type-matched approach rather
than a single universal node health signal — see Section 6 for the limitation this implies.

### Failure-type taxonomy

All 15 distinct raw `failure_type` strings (confirmed identical and exhaustive across both
the 300 training and 241 test events — no unseen variants) are mapped to a fixed English
taxonomy in `scripts/preprocessing/aiops2022_failure_taxonomy.py`:

| Raw (Chinese) | English | Scope |
|---|---|---|
| k8s容器cpu负载 | `pod_cpu_load` | pod |
| k8s容器内存负载 | `pod_memory_load` | pod |
| k8s容器读io负载 | `pod_io_read_load` | pod |
| k8s容器写io负载 | `pod_io_write_load` | pod |
| k8s容器网络丢包 | `pod_network_packet_loss` | pod |
| k8s容器网络资源包损坏 | `pod_network_packet_corruption` | pod |
| k8s容器网络资源包重复发送 | `pod_network_packet_duplication` | pod |
| k8s容器网络延迟 | `pod_network_latency` | pod |
| k8s容器进程中止 | `pod_process_abort` | pod |
| node节点CPU故障 | `node_cpu_fault` | node |
| node节点CPU爬升 | `node_cpu_climb` | node |
| node 磁盘读IO消耗 | `node_disk_read_io_consumption` | node |
| node 磁盘写IO消耗 | `node_disk_write_io_consumption` | node |
| node 磁盘空间消耗 | `node_disk_space_consumption` | node |
| node 内存消耗 | `node_memory_consumption` | node |

The loader raises an error on any unrecognized string rather than silently passing through
untranslated Chinese, so this table is guaranteed exhaustive for any future rerun.

## 3. Detection and recovery definitions

All four analysis scripts (training/test × service/node) call one shared function,
`detect_and_recover_sigma()` in `scripts/preprocessing/aiops2022_sigma_detection.py`, rather
than maintaining separate detection logic per script. This was a deliberate refactor (see
Section 5) after an audit found the original training scripts implemented a different,
undocumented method (fixed absolute/relative thresholds, not sigma-based) that had silently
diverged from the method used for the test split.

**Definitions (apply identically to both splits and both signal types):**

- **Baseline window**: the 10 minutes immediately preceding the injection timestamp
  (`[t₀ − 600s, t₀)`).
- **Baseline statistic**: per relevant value column (`rr`, `sr`, `mrt` for service-level;
  `value` for each matched node KPI), the **mean** and **standard deviation** over the
  baseline window.
- **Minimum baseline requirement**: at least 3 samples in the baseline window are required
  to compute a standard deviation; events with fewer are excluded from detection entirely
  and marked `mapped_but_no_baseline_data` rather than scored against a degenerate baseline.
  This excluded **4 training service-level events** (all `pod_network_packet_corruption`,
  all occurring too close to the start of their cloudbed/day file's timestamp range to have
  10 minutes of prior history) and **5 test service-level events** (all on the first day of
  their respective cloudbed's window, same cause). No node-level events were excluded on
  this basis in either split (all 85 training / 67 test node events had sufficient baseline
  data).
- **Detectable**: at least one sample in the 5 minutes immediately following injection
  (`[t₀, t₀ + 300s]`) deviates from the baseline mean by more than 3 standard deviations
  (`|x − μ| > 3σ`), in any one of the relevant value columns. A floor of `σ_eff = max(σ, 1e-6)`
  is applied to avoid division-by-near-zero false positives on columns that are exactly flat
  in the baseline (e.g. `rr`/`sr` sitting at 100.0 with zero variance).
- **Recovery time**: once detected, the post-injection series (scanned up to 60 minutes,
  `[t₀, t₀ + 3600s]`) is checked against a **looser** threshold — `|x − μ| > 2σ` — and
  recovery time is the timestamp of the first point that begins a run of **at least 3
  consecutive samples** all within this looser band, minus `t₀`. Using a looser threshold
  for recovery than for detection (3σ to trigger, 2σ to confirm settled) creates hysteresis,
  avoiding a series that hovers exactly at the detection boundary from flapping between
  "recovered" and "not recovered" sample to sample.
- **Non-detected / non-recovered events use `NaN`, never `0.0`.** An earlier version of the
  training pipeline used `recovery_seconds = 0.0` as a placeholder for "never detected,"
  which then leaked into the aggregate statistics: because `pandas.median()`/`.quantile()`
  do not automatically exclude explicit zeros the way they exclude `NaN`, fault types with a
  low detection rate had their reported recovery time dragged toward zero by the
  never-detected events. The clearest case was `pod_network_packet_duplication`: with 14 of
  22 training events undetected, the placeholder-zero version reported a median recovery of
  **0 seconds**, even though every one of the 8 actually-detected events had a real recovery
  time of 19–57 seconds. Switching the placeholder to `NaN` — which pandas' aggregation
  functions skip by default — corrected the reported median to **33 seconds**, computed only
  over genuinely detected-and-resolved events. This fix is applied throughout; any
  `recovery_seconds` value in the current result files is either a real elapsed time or
  `NaN`, never a sentinel zero.

## 4. Compound event flag

An event is flagged `compound = True` if **another event affecting the same resolved
signal** — the same base service name for pod/service-level events, or the same node
`cmdb_id` for node-level events — has an injection timestamp within the 60-minute
recovery/scan window (`|Δt| ≤ 3600s`) of this event's timestamp. Events with no such
neighbor are `compound = False` ("isolated"). The comparison is scoped within a single
cloudbed/date group (training) or date group (test); cross-date overlaps cannot occur given
the multi-day/multi-week gaps between groundtruth files.

This matters because a compound event's measured "recovery time" may actually reflect the
second fault's onset rather than genuine settling — the recommendationservice case found
during exploratory analysis (two separate faults on the same service ~4 minutes apart)
motivated adding this flag.

This flag applies to **both splits** with an identical definition — `compound` per event and
`n_compound` per fault type are present in all eight result files (training and test, both
signal types). Training was regenerated to backfill this column after it was initially added
to the test pipeline only; the pre-backfill training files are preserved as
`*_v3_precompound.csv` (see Section 5).

**Overall compound-vs-isolated rates:**

| | Service-level | Node-level |
|---|---|---|
| Train | 45/211 compound (21.3%) | 12/85 compound (14.1%) |
| Test | 46/169 compound (27.2%) | 10/67 compound (14.9%) |

Node-level compound rates are close between splits (~14–15%). Service-level is somewhat
higher in test (27.2% vs. 21.3%), plausibly a function of test's smaller per-date event pools
packing faults closer together in relative terms — with n this small, a 6-point gap should
not be over-interpreted as a systematic difference between the splits.

## 5. Version history

Each regeneration of the training result files was triggered by a specific, identified bug
or methodology gap, and the prior version was preserved before overwriting rather than
discarded. All backup files live alongside the current ones in `results/aiops2022/`.

| Stage | Trigger | Training service-level mapped / excluded | Preserved as |
|---|---|---|---|
| v1 (original Task 1) | Initial implementation | 196 mapped; 85 unmapped (node, no topology); 19 unmapped (unknown service prefix — shadow-replica `cmdb_id`s not yet resolved) | `aiops2022_train_event_detectability_v1_preremap.csv`, `cloudbed_detectability_by_faulttype_v1_preremap.csv` |
| v2 (shadow-replica fix) | User sanity-check request: rerun the 19 unmapped pod events with `{service}2-{instance}` resolution | 215 mapped (+19); 85 unmapped (node only) — but still used the original fixed-threshold detection method, and still had the `recovery_seconds = 0.0` placeholder bug for non-detected events | `aiops2022_train_event_detectability_v2_presharedfunc.csv`, `cloudbed_detectability_by_faulttype_v2_presharedfunc.csv` (also produced at this stage: `aiops2022_train_node_event_detectability_v2_presharedfunc.csv`, `node_detectability_by_faulttype_v2_presharedfunc.csv`, and the first test-split outputs, `aiops2022_test_event_detectability_v2_presharedfunc.csv` etc.) |
| current (shared-function refactor) | Line-by-line audit found training's detection logic (median baseline, fixed absolute/relative thresholds, full 60-minute detection window, single-threshold recovery) did not match the sigma-based spec already in use for the test split (mean baseline, 3σ/5-min detect, 2σ/60-min recovery hysteresis). Both training scripts were refactored to call the same shared `detect_and_recover_sigma()` used by the test scripts | 211 mapped (−4, all `pod_network_packet_corruption` events newly excluded by the stricter ≥3-baseline-points requirement of the sigma method); 85 unmapped (node); 4 excluded (`mapped_but_no_baseline_data`) | current files: `aiops2022_train_event_detectability.csv`, `cloudbed_detectability_by_faulttype.csv`, `aiops2022_train_node_event_detectability.csv`, `node_detectability_by_faulttype.csv`, `aiops2022_test_event_detectability.csv`, `aiops2022_test_detectability_by_faulttype.csv`, `aiops2022_test_node_event_detectability.csv`, `test_node_detectability_by_faulttype.csv` |

The test-split node/service counts (169 mapped + 5 no-baseline-data service-level; 67 mapped
node-level) did not change between the "v2" and "current" stages, because the test scripts
already called the sigma-based method from the start — only the training scripts' internal
logic changed when they were pointed at the same shared function.

**Scripts** (all under `scripts/`):

- `preprocessing/aiops2022_archive_readers.py` — streams `metric_service.csv` and
  `metric/node` CSVs directly out of the zip/7z archives (no full extraction)
- `preprocessing/aiops2022_groundtruth_loader.py` — loads the 300 training / 241 test events
  into a unified table
- `preprocessing/aiops2022_failure_taxonomy.py` — the 15-entry Chinese→English mapping
- `preprocessing/aiops2022_sigma_detection.py` — the single shared detect/recover function
  (Section 3) used by all four analysis scripts below
- `analysis/aiops2022_detectability_recovery.py` — training, service-level (also defines the
  shadow-replica `cmdb_id` resolution reused by the test service-level script)
- `analysis/aiops2022_node_detectability.py` — training, node-level (also defines
  `NODE_FAULT_KPIS`, reused by the test node-level script)
- `analysis/aiops2022_test_detectability_recovery.py` — test, service-level
- `analysis/aiops2022_test_node_detectability.py` — test, node-level

## 6. Known limitations

- **Small n per fault type per split.** Mapped event counts per fault type range from 7
  (`node_cpu_climb`, test) to 27 (`pod_cpu_load`/`pod_network_packet_corruption`, train).
  With n this small, a difference of a few events changes the detection rate by several
  percentage points, and a single slow-recovering event can shift a reported median
  noticeably. Fault-type-level swings between train and test (e.g.
  `pod_network_packet_duplication`'s recovery time moving from 108s to 604s on n=3/n=2
  detected events respectively) should be read as noisy, not as evidence of systematic
  temporal drift, without a larger sample or a variance estimate to accompany the point
  statistic.
- **Node-level detection uses a fault-type-matched KPI set, not a universal signal.** Each
  node fault type is checked only against the 2–3 KPIs listed in Section 2's table, chosen
  because they are the metrics most directly related to the injected fault mechanism (e.g.
  disk read IOPS/latency for a disk-read-IO fault). This means node-level detectability
  numbers answer "is this fault visible in the metric it's expected to perturb," not "is
  this fault visible in node telemetry at all" — a fault that manifests in an unexpected KPI
  (e.g. a CPU fault that shows up as elevated I/O wait before CPU usage itself moves) could
  be under-counted as undetected here. Unlike the service-level signal (a fixed, small set
  of business KPIs checked for every pod/service fault regardless of type), the node-level
  approach was deliberately narrowed for tractability and has not been validated against a
  broader sweep of all ~55 available node KPIs.
- **85 training / 67 test node-level events, evaluated only via `metric/node`.** These are
  reported in entirely separate result files from the service-level events (never merged
  into one table), because they are measured against a structurally different signal type
  and are not directly comparable to service-level detection rates or recovery times without
  that caveat attached.
- **Compound-event flag is test-split only** (Section 4) — training result files lack this
  column, so any train/test comparison that wants to condition on isolated-vs-compound
  events cannot currently do so on the training side.
- **Detection/recovery thresholds (3σ / 2σ / 5 min / 60 min / 3 samples) are fixed choices,
  not fit or validated against ground truth.** They were specified directly rather than
  tuned; no sensitivity analysis (e.g. how detection rate changes under 2σ vs 4σ) has been
  run. If the paper reports these detectability numbers as a primary result rather than as
  descriptive/exploratory statistics, a threshold-sensitivity check would strengthen the
  claim. (Addressed by the check in Section 7.)

## 7. Robustness check: detection-threshold sensitivity

To test whether the fault-type detection-rate ranking depends on the specific 3σ detection
threshold, the training-set classification was rerun at **2.5σ** and **4σ** in place of 3σ,
keeping the baseline window, 5-minute detection window, and mapping logic unchanged. This
was a detection-rate-only rerun (no recovery-time recomputation) using the same
`detect_and_recover_sigma()` function with its threshold overridden, and does not modify or
replace the four primary result files — output is a standalone file,
`results/aiops2022/aiops2022_train_sigma_robustness.csv` (script:
`scripts/analysis/aiops2022_train_sigma_robustness.py`).

**Ranking is largely stable.** Spearman rank correlation between the three thresholds:

| Comparison | Rank correlation |
|---|---|
| 2.5σ vs 3σ | 0.950 |
| 3σ vs 4σ | 0.953 |
| 2.5σ vs 4σ | 0.904 |

The easiest and hardest fault types hold across all three thresholds. `node_cpu_fault` and
`node_cpu_climb` are detected at 100% regardless of threshold. `pod_network_packet_duplication`
remains the single hardest-to-detect fault type at every threshold (22.7% at 2.5σ → 13.6% at
3σ → 4.5% at 4σ) — consistent with every other check on this fault type throughout this
analysis. `pod_io_read_load` stays in the bottom three throughout as well.

The one fault type whose rank moves meaningfully is `node_disk_write_io_consumption`, which
drops from the single easiest fault type at 2.5σ (100% detected) to 7th at 4σ (83.3%) as the
threshold tightens — the largest rank swing observed. A handful of other pairs swap adjacent
ranks (e.g. `pod_network_packet_loss` vs. `node_disk_read_io_consumption`), consistent with
the small-n caveat in Section 6 rather than a substantive reordering.

### 7.1 Why `node_disk_write_io_consumption` is unusually threshold-sensitive

Because the detection threshold is `sigma × σ` in absolute terms, a fault type's sensitivity
to the sigma choice depends on how noisy its own baseline already is — the same multiplier
increase inflates the required deviation far more on a high-variance baseline than a tight
one. To check this directly, baseline mean, baseline standard deviation, and coefficient of
variation (CV = σ / mean) were computed per event for `node_disk_write_io_consumption` and
two comparison fault types, `node_cpu_fault` and `node_disk_read_io_consumption` (script:
`scripts/analysis/aiops2022_node_baseline_variance_check.py`; per-event data:
`results/aiops2022/aiops2022_node_baseline_variance_by_faulttype.csv`, 108 rows).

| failure_type | KPI | median baseline mean | median baseline σ | median CV | p75 CV |
|---|---|---|---|---|---|
| `node_cpu_fault` | `system.cpu.pct_usage` | 6.68 | 1.29 | 0.27 | 0.46 |
| `node_cpu_fault` | `system.load.1` | 1.16 | 0.40 | 0.44 | 0.58 |
| `node_disk_read_io_consumption` | `r_await` / `r_s` / `rkb_s` | 0.000 | 0.000 | ~2.4 | ~3.2 |
| `node_disk_write_io_consumption` | `system.io.w_await` | 0.515 | 0.346 | 1.63 | 2.95 |
| `node_disk_write_io_consumption` | `system.io.w_s` | 22.0 | 22.0 | 1.50 | 2.49 |

**`node_disk_write_io_consumption`'s baseline is 4–6× noisier than `node_cpu_fault`'s** by
coefficient of variation (~1.5–1.6 vs. ~0.27–0.44). CPU usage sits at a low, stable
background level between fault injections, so a 3σ or 4σ band around it stays tight in
absolute terms. Write I/O, by contrast, has substantial bursty background activity even at
baseline — its median σ for `system.io.w_s` (22.0) is roughly as large as its own median
mean (22.0), meaning the "normal" range is already wide before any fault is injected. Moving
the multiplier from 2.5σ to 4σ therefore inflates the absolute detection threshold far more
for write I/O than for CPU, which is exactly why write-IO detection collapses faster as the
threshold tightens (100% → 94.4% → 83.3%) while CPU's stays flat at 100%.

**`node_disk_read_io_consumption`'s superficially similar CV (~2.4) is a different mechanism
entirely, not comparable to write I/O's.** Its baseline mean is essentially **zero** for most
events — disks are simply idle between injections — so both mean and σ round to ~0, and
dividing by a near-zero mean produces an inflated CV that reflects a numerical artifact, not
genuine noise. In this regime, detection is governed almost entirely by the
`STD_FLOOR = 1e-6` fallback in the shared detection function rather than by the baseline's
actual variance, so the effective threshold is approximately `sigma × 1e-6` — tightening
sigma from 2.5 to 4 barely moves this in absolute terms. That is why read-IO's decline across
thresholds is much gentler (92.9% → 85.7% → 78.6%) than write-IO's, despite both nominally
reporting a "high" CV: write I/O is genuinely noisy at baseline, while read I/O is
near-silent at baseline and its apparent noisiness is a divide-by-near-zero artifact rather
than a real signal characteristic.

**Conclusion**: the qualitative claim "packet duplication is dramatically harder to detect
than other pod-level fault types, and CPU-related node faults are the easiest" is robust to
the specific choice of 3σ. The exact detection-rate percentages are, unsurprisingly,
threshold-dependent and should not be over-quoted to more precision than the threshold
choice supports — but the ranking a paper would actually want to make claims about (which
fault types are hard vs. easy to detect) does not hinge on 3σ specifically over 2.5σ or 4σ.

### 7.2 Compound-status check: `pod_network_packet_duplication`

`pod_network_packet_duplication` is the single hardest-to-detect fault type throughout this
analysis, and its training-split compound events showed a striking pattern: 0 of 9 compound
events were detected, versus 3 of 13 (23.1%) isolated events. This raised the hypothesis that
compound status — a second fault landing on the same service within the 60-minute recovery
window (Section 4) — actively suppresses detection for this fault type, e.g. by
contaminating the baseline or post-injection window with a second event's effect.

This hypothesis did not survive checking it against the test split or against comparison
fault types. In the test split, the pattern reversed on a small sample: compound
packet-duplication events detected *more* often (25.0%, 1 of 4) than isolated ones (5.3%, 1
of 19). Two comparison fault types — `pod_cpu_load` and `pod_network_packet_loss`, chosen for
contrast because they combine low compound rates with high overall detection — showed
**compound ≥ isolated detection rate in every split checked** (`pod_cpu_load`: 100% vs. 100%
train, 100% vs. 92.9% test; `pod_network_packet_loss`: 100% vs. 87.0% train, 100% vs. 80.0%
test). If compound status generally suppressed detectability, these contrast cases would be
expected to show the same direction of effect, and instead they consistently point the other
way.

**Compound status is therefore not treated as an explanation for
`pod_network_packet_duplication`'s low detectability.** The training-split 0/9 pattern that
motivated the check is most plausibly a small-sample artifact rather than a real mechanism —
neither the test split nor the two comparison fault types replicate it, and the comparison
fault types' consistent compound-favoring pattern argues against a general
compound-suppresses-detection effect existing at all. The low detectability of packet
duplication remains attributable to the fault type's intrinsic characteristics — packet
duplication does not reliably perturb `rr`/`sr`/`mrt` the way packet loss, corruption, or
resource-exhaustion faults do — rather than to confounding from overlapping fault windows.
