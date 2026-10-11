export const SIZES = ['100k', '500k', '1M', '3M', '6M', '24M'];
export const PARAMETERS = { '100k': 65728, '500k': 492160, '1M': 984192, '3M': 3016960, '6M': 6033664, '24M': 24253184 };
export const MATCHED = [1.617, 3.228, 6.449, 12.891, 17.723];
export const FINAL = 17.723;
export const ARMS = {
  sp: { name: 'Self-play', color: '#D55E00', dash: '', marker: 'circle' },
  dclm: { name: 'DCLM text', color: '#0072B2', dash: '6 4', marker: 'square' },
  up: { name: 'Universal prior', color: '#777777', dash: '2 4', marker: 'triangle' },
  sp2dclm: { name: 'Self-play → text', color: '#009E73', dash: '', marker: 'circle' },
};
export const SUITES = {
  text: { name: 'Printable text', title: 'Procedural tasks', detail: 'Copying, comparison, stack tracking, recall, addition, reversal and cipher. Nine tasks, m = 32–128.', tasks: ['first', 'last', 'max', 'min', 'stack', 'assoc', 'sum', 'reverse', 'cipher'] },
  word: { name: 'Words', title: 'Lookup & word meaning', detail: 'Unseen-word classification with 2 or 4 labels and seen-word lookup. Three tasks at m = 32. Word reversal is shown separately.', tasks: ['classify-2', 'classify-4', 'lookup-4', 'word-reverse'] },
  raw: { name: 'Raw bytes', title: 'The paper’s byte format', detail: 'Random bytes 1–255, separated by byte 0. Eight tasks, m = 32–128. Raw reversal uses per-byte accuracy; the other tasks use exact match.', tasks: ['first', 'last', 'max', 'min', 'stack', 'assoc', 'sum', 'reverse'] },
};
export const TASK_NAMES = { first: 'First letter / byte', last: 'Last letter / byte', max: 'Maximum', min: 'Minimum', stack: 'Stack tracking', assoc: 'Key → value recall', sum: 'Addition', reverse: 'Sequence reversal', cipher: 'Letter cipher', 'classify-2': 'Unseen-word category · 2 labels', 'classify-4': 'Unseen-word category · 4 labels', 'lookup-4': 'Seen-word lookup · 4 labels', 'word-reverse': 'Word-order reversal' };
const CURVE_TASKS = { 'classify (unseen word, 2 labels)': 'classify-2', 'classify (unseen word, 4 labels)': 'classify-4', 'lookup (seen word, 4 labels)': 'lookup-4', 'reverse word order': 'word-reverse' };
const cache = new Map();
export function decodeTable(table) {
  return table.rows.map(values => Object.fromEntries(table.columns.map((key, i) => [key, table.dictionaries[key]?.[values[i]] ?? values[i]])));
}
export async function loadTable(name) {
  if (!cache.has(name)) cache.set(name, fetch(`./data/${name}.json`).then(response => {
    if (!response.ok) throw new Error(`Could not load ${name} (${response.status}).`);
    return response.json();
  }).then(decodeTable).then(rows => rows.map(r => ({ ...r, task: CURVE_TASKS[r.task] ?? r.task }))));
  try { return await cache.get(name); } catch (error) { cache.delete(name); throw error; }
}
export function select(rows, filters) {
  return rows.filter(row => Object.entries(filters).every(([key, value]) => value == null || row[key] === value));
}
export function summary(values) {
  if (!values.length) return null;
  const n = values.length;
  const mean = values.reduce((sum, v) => sum + v, 0) / n;
  if (n < 2) return { mean, n, low: null, high: null };
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1);
  const critical = { 1: 12.706, 2: 4.303, 3: 3.182 }[n - 1];
  if (!critical) throw new Error(`Unsupported seed count: ${n}`);
  const half = critical * Math.sqrt(variance / n);
  return { mean, n, low: mean - half, high: mean + half };
}
export function groupSeries(rows, xKey, { seed = 'all', valueKey = 'value', interval = true } = {}) {
  const grouped = new Map();
  for (const row of rows) {
    if (seed !== 'all' && row.seed !== seed) continue;
    const key = row[xKey];
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  return [...grouped].map(([x, items]) => {
    const stats = summary(items.map(r => r[valueKey]));
    return { x, y: stats.mean, low: interval ? stats.low : null, high: interval ? stats.high : null, n: stats.n, rows: items };
  }).sort((a, b) => a.x - b.x);
}
export function roundTokens(arm, size, target) {
  // UP uses 1024 or 2048 programs per round; SP uses 1536. Same round ≠ same tokens.
  if (arm !== 'up') return target;
  const programs = ['500k', '3M'].includes(size) ? 2048 : 1024;
  const rounds = { 1.617: 256, 3.228: 512, 6.449: 1024, 12.891: 2048, 17.723: 2816 };
  return Math.round(programs * 4096 * (rounds[target] + 1) / 1e6) / 1000;
}
export const formatValue = value => value == null ? '—' : value.toFixed(3);
export const formatTokens = value => value.toLocaleString('en-US', { maximumFractionDigits: 3 });
