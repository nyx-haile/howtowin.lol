<script lang="ts">
	import { clock } from '../replay';

	export let series: { t: number; diff: number }[];
	export let from: number;
	export let to: number;

	const W = 600;
	const H = 150;
	const PAD = 28;

	$: maxAbs = Math.max(1000, ...series.map((p) => Math.abs(p.diff)));
	$: tMax = series[series.length - 1]?.t ?? 1;
	$: xs = (t: number) => PAD + (t / tMax) * (W - PAD * 2);
	$: ys = (d: number) => H / 2 - (d / maxAbs) * (H / 2 - 16);
	$: path = series.map((p, i) => `${i ? 'L' : 'M'} ${xs(p.t)} ${ys(p.diff)}`).join(' ');
	$: at = (t: number) => {
		const before = [...series].reverse().find((p) => p.t <= t) ?? series[0];
		const after = series.find((p) => p.t >= t) ?? series[series.length - 1];
		if (after.t === before.t) return before.diff;
		return before.diff + ((after.diff - before.diff) * (t - before.t)) / (after.t - before.t);
	};
	$: d0 = Math.round(at(from) / 10) * 10;
	$: d1 = Math.round(at(to) / 10) * 10;
	const signed = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toLocaleString()}`;
</script>

<figure class="gold">
	<figcaption>
		<span>Team gold lead</span>
		<span class="swing">
			{clock(from)}
			<b class:up={d0 >= 0}>{signed(d0)}</b>
			→ {clock(to)} <b class:up={d1 >= 0}>{signed(d1)}</b>
		</span>
	</figcaption>
	<svg viewBox="0 0 {W} {H}" role="img" aria-label="Gold lead over time">
		<rect x={xs(from)} y="4" width={Math.max(2, xs(to) - xs(from))} height={H - 8} class="window" />
		<line x1={PAD} x2={W - PAD} y1={H / 2} y2={H / 2} class="zero" />
		<path d={path} class="line" />
		<circle cx={xs(from)} cy={ys(at(from))} r="5" class="dot" />
		<circle cx={xs(to)} cy={ys(at(to))} r="5" class="dot end" />
		<text x={PAD} y={H - 6} class="axis">0:00</text>
		<text x={W - PAD} y={H - 6} class="axis end">{clock(tMax)}</text>
		<text x={PAD} y="14" class="axis">ahead</text>
		<text x={PAD} y={H - 20} class="axis">behind</text>
	</svg>
</figure>

<style>
	.gold {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	figcaption {
		display: flex;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 6px;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.swing b {
		color: var(--loss);
		font-variant-numeric: tabular-nums;
	}
	.swing b.up {
		color: var(--win);
	}
	svg {
		width: 100%;
		height: auto;
	}
	.window {
		fill: color-mix(in srgb, var(--gold) 14%, transparent);
	}
	.zero {
		stroke: var(--border);
		stroke-width: 1;
	}
	.line {
		fill: none;
		stroke: var(--text);
		stroke-width: 2;
	}
	.dot {
		fill: var(--gold);
	}
	.dot.end {
		fill: var(--text);
	}
	.axis {
		font-size: 10px;
		fill: var(--text-muted);
	}
	.axis.end {
		text-anchor: end;
	}
</style>
