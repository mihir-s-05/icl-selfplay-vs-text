import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { decodeTable, select, groupSeries, summary, roundTokens, SIZES, FINAL, MATCHED } from '../src/data.js';

const dataURL = name => new URL(`../dist/data/${name}`, import.meta.url);
const read = name => decodeTable(JSON.parse(readFileSync(dataURL(`${name}.json`))));
const H = read('headline');
function parseLine(line) {
  const fields = []; let field = ''; let quoted = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === '"') {
      if (quoted && line[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (line[i] === ',' && !quoted) { fields.push(field); field = ''; }
    else field += line[i];
  }
  fields.push(field); return fields;
}

test('all released tables retain their complete row counts, checksums and numeric precision', () => {
  const manifest = JSON.parse(readFileSync(dataURL('manifest.json')));
  assert.equal(manifest.sourceCommit, '74e8f7be101e050c6982f2838f2d3ea980851f5f');
  const expected = { headline: 1191, tasks: 8337, mcurves: 78209, valbpb: 1248, sp_bpb: 198 };
  for (const [name, count] of Object.entries(expected)) {
    const csv = readFileSync(dataURL(`${name}.csv`));
    assert.equal(createHash('sha256').update(csv).digest('hex'), manifest.tables[name].sha256);
    const [header, ...lines] = csv.toString().trim().split(/\r?\n/).map(parseLine);
    const rows = read(name);
    assert.equal(rows.length, count);
    assert.equal(lines.length, count);
    // Compare every decoded cell, including comma-containing task names.
    rows.forEach((row, i) => header.forEach((key, j) => assert.equal(String(row[key]), typeof row[key] === 'number' ? String(Number(lines[i][j])) : lines[i][j])));
  }
});

test('the headline figure reproduces released 24M means and differences', () => {
  const mean = (arm, suite) => summary(select(H, { arm, suite, size: '24M', tokens: FINAL }).map(r => r.value)).mean;
  assert.ok(Math.abs((mean('sp', 'text') - mean('dclm', 'text')) - .258465) < 1e-10);
  assert.ok(Math.abs((mean('dclm', 'word') - mean('sp', 'word')) - .27995) < 1e-10);
});

test('95% seed uncertainty uses Student t, keeps unclipped bounds and omits single-seed CI', () => {
  const stats = summary([0, .2, .4, .6]);
  assert.equal(stats.n, 4); assert.ok(Math.abs(stats.mean - .3) < 1e-12);
  assert.ok(Math.abs(stats.low - (.3 - 3.182 * Math.sqrt(.2 / 3 / 4))) < 1e-12);
  assert.ok(stats.low < 0);
  assert.deepEqual(summary([.5]), { mean: .5, n: 1, low: null, high: null });
  assert.equal(summary([]), null);
});

test('the baseline is compared at the same round using its actual unequal token counts', () => {
  for (const size of SIZES.filter(size => size !== '24M')) {
    for (const target of MATCHED) {
      const tokens = roundTokens('up', size, target);
      const availableSeeds = size === '100k' && target <= 3.228 ? 2 : 4;
      assert.equal(select(H, { arm: 'up', size, suite: 'text', tokens }).length, availableSeeds);
      assert.notEqual(tokens, target);
    }
  }
  assert.equal(select(H, { arm: 'up', size: '24M' }).length, 0);
});

test('exact seed filters preserve the text arm’s limited and unequal coverage', () => {
  assert.equal(select(H, { arm: 'dclm', size: '24M', suite: 'word', seed: 'seed-1' }).length, 0);
  const means = select(H, { size: '6M', suite: 'word', tokens: FINAL });
  const runs = groupSeries(select(means, { arm: 'sp' }), 'tokens', { seed: 'seed-40354564' });
  assert.equal(runs.length, 1); assert.equal(runs[0].n, 1); assert.equal(runs[0].low, null);
  assert.equal(groupSeries(select(means, { arm: 'dclm' }), 'tokens', { seed: 'seed-40354564' }).length, 0);
});

test('headline suite arithmetic matches task cells, excluding only word reversal', () => {
  const tasks = read('tasks');
  for (const suite of ['text', 'word', 'raw']) for (const arm of ['sp', 'dclm']) {
    for (const row of select(H, { arm, size: '24M', suite, tokens: FINAL })) {
      const cells = select(tasks, { arm, size: '24M', suite, tokens: FINAL, seed: row.seed }).filter(r => suite !== 'word' || r.task !== 'word-reverse');
      const average = cells.reduce((sum, r) => sum + r.value, 0) / cells.length;
      assert.ok(Math.abs(average - row.value) < 1e-5);
    }
  }
  assert.ok(select(tasks, { task: 'word-reverse' }).every(row => row.value === 0));
});

test('warm-start progress is additional text tokens, with one seed and no fabricated zero point', () => {
  const rows = select(H, { arm: 'sp2dclm', size: '24M', suite: 'word' });
  assert.deepEqual([...new Set(rows.map(r => r.tokens))], [.05, .1, .25, .5]);
  assert.deepEqual([...new Set(rows.map(r => r.seed))], ['seed-0']);
  assert.ok(groupSeries(rows, 'tokens', { interval: false }).every(p => p.low === null));
});

test('text has individual observations without inferred seed intervals', () => {
  const points = groupSeries(select(H, { arm: 'dclm', size: '6M', suite: 'word' }), 'tokens', { interval: false });
  assert.ok(points.every(p => p.n === 2 && p.low === null && p.high === null));
});

test('the build contains only allowlisted static files and released artifacts', () => {
  const root = new URL('../dist/', import.meta.url);
  const files = readdirSync(root, { recursive: true }).filter(name => !name.endsWith('/')).map(name => String(name).replaceAll('\\', '/'));
  assert.ok(files.every(name => /^(data|figures)(\/|$)/.test(name) || ['index.html', 'app.js', 'chart.js', 'data.js', 'styles.css', '_headers', 'LICENSE', 'NOTICE'].includes(name)), files.join(', '));
  assert.ok(files.every(name => !/\.env|\.pth|run\.json|state|social|chat\.md|PLAN\.md/.test(name)));
});
