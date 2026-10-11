import { formatValue } from './data.js';
const NS = 'http://www.w3.org/2000/svg';
function element(name, attributes = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function marker(parent, x, y, style, hollow = false, title = '') {
  const attrs = { fill: hollow ? 'white' : style.color, stroke: style.color, 'stroke-width': 1.5 };
  const node = style.marker === 'square' ? element('rect', { x: x - 3.2, y: y - 3.2, width: 6.4, height: 6.4, ...attrs })
    : style.marker === 'triangle' ? element('path', { d: `M${x} ${y - 4}l4 7h-8Z`, ...attrs })
      : element('circle', { cx: x, cy: y, r: 3.3, ...attrs });
  if (title) node.append(element('title', {}, title));
  parent.append(node);
}
export function renderChart(host, { series, xTicks, xLabel, yLabel = 'Accuracy', yMax = 1, transform = x => x, categorical = false, compact = false, title, chance = null }) {
  host.replaceChildren();
  const width = compact ? 350 : 850;
  const height = compact ? 292 : 390;
  const margin = { left: compact ? 44 : 58, right: 16, top: 24, bottom: categorical ? 102 : 54 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const allX = xTicks.map(t => transform(t.value));
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const x = value => margin.left + ((transform(value) - minX) / (maxX - minX || 1)) * innerW;
  const y = value => margin.top + innerH * (1 - Math.max(0, Math.min(yMax, value)) / yMax);
  const svg = element('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${title}. ${yLabel} by ${xLabel}. Exact values are available in the figure data table.`, class: 'plot' });
  svg.append(element('title', {}, title));
  for (let i = 0; i <= 5; i++) {
    const value = yMax * i / 5;
    svg.append(element('line', { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), class: 'gridline' }));
    svg.append(element('text', { x: margin.left - 9, y: y(value) + 4, 'text-anchor': 'end', class: 'tick' }, value.toFixed(yMax > 2 ? 1 : 1)));
  }
  if (chance != null) {
    svg.append(element('line', { x1: margin.left, x2: width - margin.right, y1: y(chance), y2: y(chance), stroke: '#777', 'stroke-dasharray': '3 4' }));
    svg.append(element('text', { x: width - margin.right, y: y(chance) - 6, 'text-anchor': 'end', class: 'tick' }, `chance ${chance}`));
  }
  for (const tick of xTicks) {
    svg.append(element('line', { x1: x(tick.value), x2: x(tick.value), y1: y(0), y2: y(0) + 4, class: 'axisline' }));
    const tickNode = element('text', { x: x(tick.value), y: y(0) + 19, 'text-anchor': categorical ? 'end' : 'middle', class: 'tick' }, tick.label);
    if (categorical) tickNode.setAttribute('transform', `rotate(-32 ${x(tick.value)} ${y(0) + 19})`);
    svg.append(tickNode);
  }
  svg.append(element('line', { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), class: 'axisline' }));
  svg.append(element('text', { x: margin.left + innerW / 2, y: height - 7, 'text-anchor': 'middle', class: 'axislabel' }, xLabel));
  svg.append(element('text', { x: 13, y: margin.top + innerH / 2, transform: `rotate(-90 13 ${margin.top + innerH / 2})`, 'text-anchor': 'middle', class: 'axislabel' }, yLabel));
  for (const s of series) {
    const points = s.points.filter(p => Number.isFinite(p.y));
    const bandPoints = points.filter(p => p.low != null && p.high != null);
    if (bandPoints.length > 1 && !categorical) {
      const polygon = [...bandPoints.map(p => `${x(p.x)},${y(p.high)}`), ...bandPoints.toReversed().map(p => `${x(p.x)},${y(p.low)}`)].join(' ');
      svg.append(element('polygon', { points: polygon, fill: s.color, opacity: .13 }));
    }
    if (points.length > 1 && !categorical && s.connect !== false) svg.append(element('polyline', { points: points.map(p => `${x(p.x)},${y(p.y)}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': 1.8, 'stroke-dasharray': s.dash ?? '' }));
    for (const p of points) {
      if (categorical && p.low != null) {
        svg.append(element('line', { x1: x(p.x), x2: x(p.x), y1: y(p.low), y2: y(p.high), stroke: s.color, 'stroke-width': 1.4 }));
        for (const end of [p.low, p.high]) svg.append(element('line', { x1: x(p.x) - 4, x2: x(p.x) + 4, y1: y(end), y2: y(end), stroke: s.color }));
      }
      marker(svg, x(p.x), y(p.y), s, !!s.hollow, `${s.name}: ${formatValue(p.y)}; ${p.n ?? 1} seed(s)`);
      if (s.individuals) for (const row of p.rows ?? []) marker(svg, x(p.x), y(row.value), s, true, `${s.name}, ${row.seed}: ${row.value}`);
    }
  }
  if (!series.some(s => s.points.length)) svg.append(element('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle', class: 'emptyplot' }, 'No released measurements for this selection.'));
  host.append(svg);
}
