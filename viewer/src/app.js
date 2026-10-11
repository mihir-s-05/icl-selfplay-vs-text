import { ARMS, SIZES, PARAMETERS, MATCHED, FINAL, SUITES, TASK_NAMES, loadTable, select, groupSeries, roundTokens, formatValue, formatTokens } from './data.js';
import { renderChart } from './chart.js';

const $ = id => document.getElementById(id);
const state = { view: 'training', size: '24M', suite: 'text', task: 'headline', seed: 'all', checkpoint: FINAL, scaling: FINAL, prior: false };
const VIEWS = ['training', 'tasks', 'examples', 'transfer', 'loss'];
let renderVersion = 0;
let scalingExport = [];
let explorerExport = [];

function options(host, values, selected) {
  host.replaceChildren(...values.map(([value, label]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.selected = String(value) === String(selected);
    return option;
  }));
}
function legend(host, arms, individual = false) {
  host.replaceChildren(...arms.map(arm => {
    const style = typeof arm === 'string' ? ARMS[arm] : arm;
    const node = document.createElement('span');
    const swatch = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    swatch.setAttribute('viewBox', '0 0 29 12');
    const line = document.createElementNS(swatch.namespaceURI, 'line');
    for (const [key, value] of Object.entries({ x1: 0, x2: 29, y1: 6, y2: 6, stroke: style.color, 'stroke-width': 2, 'stroke-dasharray': style.dash })) line.setAttribute(key, value);
    swatch.append(line);
    node.append(swatch, `${style.name}${individual && (style === ARMS.dclm || style.arm === 'dclm') && style.individuals !== false ? ' · individual runs hollow' : ''}`);
    return node;
  }));
}
function table(host, columns, rows, captionText) {
  const result = document.createElement('table');
  const caption = document.createElement('caption');
  caption.textContent = captionText;
  result.append(caption);
  const header = result.createTHead().insertRow();
  for (const [, label] of columns) {
    const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; header.append(cell);
  }
  const body = result.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const [key] of columns) tr.insertCell().textContent = row[key] ?? '—';
  }
  host.replaceChildren(result);
}
function csvDownload(rows, name) {
  if (!rows.length) return;
  const columns = Object.keys(rows[0]);
  const quote = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const csv = [columns, ...rows.map(row => columns.map(key => row[key]))].map(row => row.map(quote).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function seriesFor(rows, arm, xKey, config = {}) {
  return { ...ARMS[arm], arm, points: groupSeries(select(rows, { arm }), xKey, { seed: state.seed, interval: arm === 'sp' || arm === 'up', ...config }), individuals: arm === 'dclm' && state.seed === 'all' };
}
function scalarSeries(rows, arm, suite, token) {
  const points = SIZES.flatMap(size => {
    const values = select(rows, { arm, suite, size, tokens: roundTokens(arm, size, token) });
    if (!values.length) return [];
    return groupSeries(values.map(row => ({ ...row, parameters: PARAMETERS[size] })), 'parameters', { interval: arm === 'sp' || arm === 'up' });
  });
  return { ...ARMS[arm], arm, points, individuals: arm === 'dclm' && $('show-runs').checked };
}
async function renderScaling() {
  const rows = await loadTable('headline');
  const token = Number($('scaling-tokens').value);
  const arms = ['sp', 'dclm', ...($('show-up').checked ? ['up'] : [])];
  legend($('scaling-legend'), arms, $('show-runs').checked);
  scalingExport = [];
  for (const suite of ['text', 'word', 'raw']) {
    const series = arms.map(arm => scalarSeries(rows, arm, suite, token));
    renderChart($(`scaling-${suite}`), { series, xTicks: [65728, 984192, 6033664, 24253184].map((value, i) => ({ value, label: ['65.7k', '984k', '6.03M', '24.25M'][i] })), transform: Math.log10, xLabel: 'Non-embedding parameters (log)', yLabel: suite === 'raw' ? 'Mean task accuracy' : 'Mean exact-match accuracy', title: SUITES[suite].title, compact: true });
    for (const arm of arms) for (const size of SIZES) {
      const selected = select(rows, { arm, suite, size, tokens: roundTokens(arm, size, token) });
      const point = groupSeries(selected.map(r => ({ ...r, x: 0 })), 'x', { interval: arm === 'sp' || arm === 'up' })[0];
      if (point) scalingExport.push({ format: suite, arm, size, non_embedding_parameters: PARAMETERS[size], learner_tokens_B: selected[0].tokens, mean: point.y, seed_count: point.n, seed_ci_low: point.low, seed_ci_high: point.high, per_seed: selected.map(r => `${r.seed}=${r.value}`).join('; ') });
    }
  }
  $('scaling-caption').replaceChildren();
  const label = document.createElement('strong'); label.textContent = 'Figure 1. ';
  $('scaling-caption').append(label, `In-context learning by model size at ${formatTokens(token)}B learner tokens. Self-play: mean of 4 seeds with a 95% t interval. DCLM: mean with individual text runs${$('show-runs').checked ? ' shown hollow' : ' available in the table'} (1–2 seeds per size). All accuracy axes span 0–1. Raw reversal uses per-byte scoring; other tasks use exact match. `,
    $('show-up').checked ? 'Universal prior is shown at the corresponding round, using ⅔ or 4/3 of the SP learner tokens (see table); it is not token-matched. UP has 2–4 available seeds and no released 24M checkpoint.' : 'Word-order reversal is excluded from the word headline; it remains available in the task explorer.');
  const getMean = (arm, suite) => scalingExport.find(r => r.arm === arm && r.format === suite && r.size === '24M')?.mean;
  const proceduralGap = getMean('sp', 'text') - getMean('dclm', 'text');
  const wordGap = getMean('dclm', 'word') - getMean('sp', 'word');
  $('scaling-findings').replaceChildren(...[
    `${proceduralGap >= 0 ? 'Self-play leads' : 'Text leads'} by ${Math.abs(proceduralGap).toFixed(3)} on the printable-text suite at 24M. The per-task view shows which abilities drive the mean.`,
    `${wordGap >= 0 ? 'Text leads' : 'Self-play leads'} by ${Math.abs(wordGap).toFixed(3)} on the word suite at 24M. Prompt format changes what the comparison measures.`,
  ].map(text => { const p = document.createElement('p'); p.textContent = text; return p; }));
  const displayRows = scalingExport.map(row => ({ ...row, mean: formatValue(row.mean), seed_ci_low: formatValue(row.seed_ci_low), seed_ci_high: formatValue(row.seed_ci_high) }));
  table($('scaling-table'), [['format', 'Format'], ['arm', 'Arm'], ['size', 'Size'], ['learner_tokens_B', 'Learner tokens (B)'], ['mean', 'Mean'], ['seed_count', 'Seeds'], ['seed_ci_low', '95% CI low'], ['seed_ci_high', '95% CI high'], ['per_seed', 'Actual seed values']], displayRows, 'Means and unclipped intervals displayed to 3 decimals; downloads retain full precision. DCLM has no displayed seed interval.');
}

function updateControls() {
  for (const button of document.querySelectorAll('[data-view]')) {
    const active = button.dataset.view === state.view;
    button.setAttribute('aria-selected', active); button.tabIndex = active ? 0 : -1;
  }
  $('explorer-panel').setAttribute('aria-labelledby', `tab-${state.view}`);
  const loss = state.view === 'loss';
  $('suite-control').hidden = loss;
  $('task-control').hidden = loss || state.view === 'tasks';
  $('checkpoint-control').hidden = !['tasks', 'examples'].includes(state.view);
  $('prior-control').hidden = ['transfer', 'loss'].includes(state.view);
  const taskOptions = SUITES[state.suite].tasks.map(task => [task, TASK_NAMES[task]]);
  if (state.view !== 'examples') taskOptions.unshift(['headline', 'Suite mean']);
  if (!taskOptions.some(([key]) => key === state.task)) state.task = state.view === 'examples' ? SUITES[state.suite].tasks[0] : 'headline';
  options($('task'), taskOptions, state.task);
  $('model-size').value = state.size; $('suite').value = state.suite; $('checkpoint').value = state.checkpoint;
  $('explorer-up').checked = state.prior;
}
function updateURL() {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state)) params.set(key, value);
  history.replaceState(null, '', `?${params}${location.hash}`);
}
function metadataRows(series, xLabel) {
  return series.flatMap(s => s.points.flatMap(p => (p.rows ?? []).map(row => ({
    arm: s.arm ?? 'sp', series: s.name, size: state.size, format: state.view === 'loss' ? 'DCLM text loss' : state.suite,
    task: state.view === 'tasks' ? row.task : state.task, seed: row.seed ?? 'paper K=1 mean (seed IDs unavailable)',
    x_axis: xLabel, x_value: state.view === 'tasks' ? row.task : p.x, learner_or_additional_text_tokens_B: row.tokens,
    ...(state.view === 'transfer' ? { prior_selfplay_learner_tokens_B: s.arm === 'sp2dclm' ? 51.54 : 0 } : {}),
    value: row.value ?? row.val_bpb ?? row.dclm_bpb,
    plotted_mean: p.y, seed_count: p.n, seed_ci_low: p.low, seed_ci_high: p.high,
  }))));
}
async function renderExplorer() {
  const version = ++renderVersion;
  updateControls(); updateURL();
  $('data-error').hidden = true;
  $('explorer-panel').setAttribute('aria-busy', 'true');
  $('selection-status').textContent = 'Loading released measurements…';
  try {
    let sourceRows, series, xTicks, xLabel, title, context, caption;
    let transform = Math.log10, yMax = 1, categorical = false, chance = null;
    const arms = ['sp', 'dclm', ...(state.prior && !['transfer', 'loss'].includes(state.view) ? ['up'] : [])];
    const base = { size: state.size, suite: state.suite };
    if (state.view === 'loss') {
      const [loss, zeroShot] = await Promise.all([loadTable('valbpb'), loadTable('sp_bpb')]);
      if (version !== renderVersion) return;
      sourceRows = select(loss, { size: state.size });
      series = ['dclm', 'sp2dclm'].flatMap(arm => {
        const rows = select(sourceRows, { arm });
        const seeds = [...new Set(rows.map(r => r.seed))].filter(seed => state.seed === 'all' || seed === state.seed);
        return seeds.map(seed => ({ ...ARMS[arm], arm, name: `${ARMS[arm].name} · ${seed}`, individuals: false, points: groupSeries(select(rows, { seed }), 'tokens', { valueKey: 'val_bpb', interval: false }) }));
      });
      if (state.seed === 'all') series.push({ ...ARMS.sp, arm: 'sp', name: 'Self-play · paper zero-shot mean', points: groupSeries(select(zeroShot, { size: state.size }), 'tokens', { valueKey: 'dclm_bpb', interval: false }) });
      xTicks = [.006, .1, 1, 10, 60].map(value => ({ value, label: String(value) }));
      xLabel = 'Learner tokens (B, log); warm start: additional text tokens';
      yMax = Math.max(10, ...series.flatMap(s => s.points.map(p => p.y)));
      title = `Text modelling loss · ${state.size}`;
      context = 'Validation bits per byte ↓ lower is better. This is a different metric from ICL accuracy.';
      caption = 'Text and warm-start curves are individual validation runs on held-out DCLM. Self-play is the original paper’s K=1 zero-shot DCLM mean, with no per-seed table. Warm-start x excludes 51.54B prior SP learner tokens. The zero-shot and validation settings are documented in the source; no uncertainty interval is inferred.';
    } else if (state.view === 'tasks') {
      const rows = await loadTable('tasks');
      if (version !== renderVersion) return;
      sourceRows = select(rows, base);
      const tasks = SUITES[state.suite].tasks;
      series = arms.map(arm => {
        const filtered = select(sourceRows, { arm, tokens: roundTokens(arm, state.size, state.checkpoint) }).map(row => ({ ...row, position: tasks.indexOf(row.task) }));
        return { ...seriesFor(filtered, arm, 'position'), points: seriesFor(filtered, arm, 'position').points.map(p => ({ ...p, x: p.x + (arm === 'sp' ? -.12 : arm === 'dclm' ? 0 : .12) })) };
      });
      xTicks = tasks.map((task, i) => ({ value: i, label: ({ 'classify-2': 'Category · 2', 'classify-4': 'Category · 4', 'lookup-4': 'Word lookup', 'word-reverse': 'Word reversal', assoc: 'Key → value', first: 'First', last: 'Last', reverse: 'Reversal', cipher: 'Cipher', stack: 'Stack' })[task] ?? TASK_NAMES[task] }));
      // Leave room for the offset marks at either end of the categorical axis.
      xTicks.unshift({ value: -.4, label: '' }); xTicks.push({ value: tasks.length - .6, label: '' });
      xLabel = 'Task'; transform = x => x; categorical = true;
      title = `${SUITES[state.suite].name} tasks · ${state.size} · ${formatTokens(state.checkpoint)}B tokens`;
      context = SUITES[state.suite].detail;
      caption = 'Every task is retained, including addition, reversal and cipher results near zero. Whiskers are self-play seed intervals, not intervals for the SP–text difference. Text mean and individual text seeds are shown. The fixed example counts and task cells match the published extraction script.';
    } else if (state.view === 'examples') {
      const rows = await loadTable('mcurves');
      if (version !== renderVersion) return;
      sourceRows = select(rows, { ...base, task: state.task });
      series = arms.map(arm => seriesFor(select(sourceRows, { arm, tokens: roundTokens(arm, state.size, state.checkpoint) }), arm, 'm'));
      const max = Math.max(1, ...series.flatMap(s => s.points.map(p => p.x)));
      xTicks = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512].filter(value => value <= max).map(value => ({ value, label: String(value) }));
      transform = x => Math.log2(x + 1);
      xLabel = 'In-context examples m (log₂(m + 1))';
      title = `${TASK_NAMES[state.task]} · ${state.size} · ${formatTokens(state.checkpoint)}B tokens`;
      context = `Prompt format: ${SUITES[state.suite].name}. Only measured example counts are plotted; no missing m = 0 result is inferred.`;
      chance = state.suite === 'word' ? ({ 'classify-2': .5, 'classify-4': .25, 'lookup-4': .25 })[state.task] ?? null : null;
      caption = 'Accuracy as worked examples accumulate. The transform includes zero explicitly. Key→value recall begins at m = 16 because the dictionary has 16 entries. Word lookup can decline as more examples add distractors. Lines connect measured points; they do not add measurements.';
    } else {
      const rows = await loadTable(state.task === 'headline' ? 'headline' : 'tasks');
      if (version !== renderVersion) return;
      sourceRows = select(rows, { ...base, ...(state.task === 'headline' ? {} : { task: state.task }) });
      if (state.view === 'transfer') {
        series = ['sp2dclm', 'dclm'].map(arm => seriesFor(sourceRows, arm, 'tokens', { interval: false }));
        xTicks = [.05, .1, .25, .5, 1, 3.228, 17.723].map(value => ({ value, label: formatTokens(value) }));
        xLabel = 'Additional DCLM text tokens (B, log)';
        title = `From self-play to text · ${state.size}`;
        context = 'Warm start: one run after 51.54B self-play learner tokens. Scratch: 1–2 text seeds. First measured warm-start point: 0.05B text tokens.';
        caption = 'The transfer run begins from the final self-play checkpoint (round 8191) and stops after 0.5B additional text tokens. Scratch training continues to 17.723B. The x-axis excludes self-play’s prior training and generator cost, so faster text acquisition is not a claim of lower total compute. Warm-start LR is 3e-3; scratch LRs differ by size. No warm-start seed interval is available.';
      } else {
        series = arms.map(arm => seriesFor(sourceRows, arm, 'tokens'));
        xTicks = [.006, .1, 1, 10, 70].map(value => ({ value, label: formatTokens(value) }));
        xLabel = 'Learner tokens (B, log)';
        title = `Training progress · ${state.size}`;
        context = `Prompt format: ${SUITES[state.suite].name}. ${state.task === 'headline' ? SUITES[state.suite].detail : TASK_NAMES[state.task]}`;
        caption = 'Each curve uses its actual released learner-token counts, including checkpoints outside the five matched SP/text counts in Figure 1. Self-play performance can vary with checkpoint and seed. Learner tokens exclude self-play generator and RL cost; they are not total compute. Text runs stop at 17.723B, self-play at 51.54B.';
      }
    }
    if (version !== renderVersion) return;
    if (state.prior && !['loss', 'transfer'].includes(state.view)) caption += ' Universal prior uses its own learner-token counts at the corresponding round; it is not token-matched and has no released 24M checkpoint.';
    $('explorer-title').textContent = title;
    $('metric-context').textContent = context;
    legend($('explorer-legend'), series.filter(s => s.points.length), state.seed === 'all');
    renderChart($('explorer-chart'), { series, xTicks, xLabel, title, transform, yMax, categorical, chance, yLabel: state.view === 'loss' ? 'Validation / zero-shot bits per byte' : state.suite === 'raw' ? 'Task accuracy (raw reversal: per-byte)' : 'Exact-match accuracy' });
    const label = document.createElement('strong'); label.textContent = 'Figure 2. ';
    $('explorer-caption').replaceChildren(label, caption);
    explorerExport = metadataRows(series, xLabel);
    const count = explorerExport.length;
    const seedIDs = [...new Set(explorerExport.map(r => r.seed))];
    const missing = series.filter(s => !s.points.length).map(s => s.name);
    $('selection-status').textContent = `${count.toLocaleString()} released measurements · ${seedIDs.length} seed / source group${seedIDs.length === 1 ? '' : 's'}.${missing.length ? ` No released data for ${missing.join(', ')} in this selection.` : ''}${state.seed !== 'all' ? ' Seed IDs identify different runs across arms; they are not paired experiments.' : ''}`;
    const tableRows = explorerExport.map(row => ({ ...row, task: TASK_NAMES[row.task] ?? row.task, plotted_mean: formatValue(row.plotted_mean), seed_ci_low: formatValue(row.seed_ci_low), seed_ci_high: formatValue(row.seed_ci_high) }));
    table($('explorer-table'), [['series', 'Series'], ['task', 'Task'], ['seed', 'Actual seed / source'], ['x_value', 'X value'], ['learner_or_additional_text_tokens_B', 'Learner / added text tokens (B)'], ['value', state.view === 'loss' ? 'Bits per byte' : 'Accuracy'], ['plotted_mean', 'Plotted mean'], ['seed_ci_low', 'CI low'], ['seed_ci_high', 'CI high']], tableRows, 'All plotted source measurements. Download retains exact values and unclipped CI bounds.');
  } catch (error) {
    if (version !== renderVersion) return;
    $('data-error').hidden = false;
    $('data-error').textContent = `${error.message} Reload the page or use the published CSV downloads below.`;
    $('selection-status').textContent = 'This selection could not be loaded.';
    $('explorer-chart').replaceChildren(); explorerExport = [];
  } finally {
    if (version === renderVersion) $('explorer-panel').setAttribute('aria-busy', 'false');
  }
}
function restoreURL() {
  const params = new URLSearchParams(location.search);
  for (const [key, allowed] of Object.entries({ view: VIEWS, size: SIZES, suite: Object.keys(SUITES), task: ['headline', ...Object.keys(TASK_NAMES)], seed: ['all', 'seed-0', 'seed-1', ...[40354564, 40354565, 40354566, 40354567].map(n => `seed-${n}`)] })) if (allowed.includes(params.get(key))) state[key] = params.get(key);
  for (const key of ['scaling', 'checkpoint']) if (MATCHED.includes(Number(params.get(key)))) state[key] = Number(params.get(key));
  state.prior = params.get('prior') === 'true';
}
async function showSources() {
  const response = await fetch('./data/manifest.json');
  if (!response.ok) throw new Error('The source manifest could not be loaded.');
  const manifest = await response.json();
  const descriptions = { headline: 'Suite scores', tasks: 'Per-task scores', mcurves: 'Accuracy vs. examples', valbpb: 'Text validation loss', sp_bpb: 'Self-play zero-shot text loss' };
  for (const [name, metadata] of Object.entries(manifest.tables)) {
    const row = document.createElement('div'); row.className = 'download-row';
    const description = document.createElement('div'); const title = document.createElement('strong'); title.textContent = descriptions[name]; const size = document.createElement('span'); size.textContent = `${metadata.rows.toLocaleString()} rows · ${name}.csv`; description.append(title, size);
    const links = document.createElement('div');
    for (const type of ['csv', 'json']) { const a = document.createElement('a'); a.href = `./data/${name}.${type}`; a.download = `${name}.${type}`; a.textContent = type.toUpperCase(); links.append(a); }
    row.append(description, links); $('downloads').append(row);
  }
  const originals = [['fig1_scaling', 'Fig. 1 · Model-size comparison'], ['fig2_tokens', 'Fig. 2 · Matched training curves'], ['fig3_tasks', 'Fig. 3 · Task differences'], ['fig4_warm', 'Fig. 4 · Transfer'], ['fig5_mcurves', 'Fig. 5 · In-context examples']];
  for (const [name, title] of originals) {
    const row = document.createElement('div'); row.className = 'original-row'; const label = document.createElement('span'); label.textContent = title; const links = document.createElement('span');
    for (const ext of ['svg', 'pdf']) { const a = document.createElement('a'); a.href = `./figures/${name}.${ext}`; a.textContent = ext.toUpperCase(); links.append(a); }
    row.append(label, links); $('original-figures').append(row);
  }
  $('provenance').append('Source snapshot: ');
  const commit = document.createElement('a'); commit.href = `${manifest.repository}/commit/${manifest.sourceCommit}`; commit.textContent = manifest.sourceCommit.slice(0, 12); $('provenance').append(commit, ' · ');
  for (const [href, label] of [['./data/manifest.json', 'Provenance & checksums'], ['./data/writeup.md', 'Download writeup']]) { const a = document.createElement('a'); a.href = href; a.textContent = label; $('provenance').append(a, ' '); }
}
function bindEvents() {
  for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => { state.view = button.dataset.view; renderExplorer(); });
  const tabs = [...document.querySelectorAll('[data-view]')];
  tabs.forEach((button, i) => button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next != null) { event.preventDefault(); tabs[next].focus(); tabs[next].click(); }
  }));
  for (const [id, key] of [['model-size', 'size'], ['suite', 'suite'], ['task', 'task'], ['seed', 'seed'], ['checkpoint', 'checkpoint']]) $(id).addEventListener('change', () => { state[key] = key === 'checkpoint' ? Number($(id).value) : $(id).value; renderExplorer(); });
  $('explorer-up').addEventListener('change', () => { state.prior = $('explorer-up').checked; renderExplorer(); });
  $('scaling-tokens').addEventListener('change', () => { state.scaling = Number($('scaling-tokens').value); updateURL(); renderScaling(); });
  for (const id of ['show-runs', 'show-up']) $(id).addEventListener('change', renderScaling);
  $('reset-view').addEventListener('click', () => { Object.assign(state, { size: '24M', suite: state.view === 'transfer' ? 'word' : 'text', task: 'headline', seed: 'all', checkpoint: FINAL, prior: false }); $('seed').value = 'all'; renderExplorer(); });
  $('download-scaling').addEventListener('click', () => csvDownload(scalingExport, 'icl-figure-1.csv'));
  $('download-explorer').addEventListener('click', () => csvDownload(explorerExport, `icl-${state.view}-${state.size}-${state.suite}.csv`));
}
async function init() {
  restoreURL();
  options($('model-size'), SIZES.map(size => [size, size]), state.size);
  for (const id of ['scaling-tokens', 'checkpoint']) options($(id), MATCHED.map(value => [value, `${formatTokens(value)}B`]), state[id === 'checkpoint' ? 'checkpoint' : 'scaling']);
  options($('seed'), [['all', 'All released seeds'], ['seed-0', 'Text / transfer · seed-0'], ['seed-1', 'Text · seed-1'], ...[40354564, 40354565, 40354566, 40354567].map(value => [`seed-${value}`, `SP / prior · ${value}`])], state.seed);
  bindEvents();
  const results = await Promise.allSettled([renderScaling(), renderExplorer(), showSources()]);
  const failed = results.find(result => result.status === 'rejected');
  if (failed) { $('data-error').hidden = false; $('data-error').textContent = failed.reason.message; }
}
init();
