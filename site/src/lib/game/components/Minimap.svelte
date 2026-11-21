<script lang="ts">
	import { onMount } from 'svelte';
	import type { GameEvent, GameReplay, Side } from '../types';
	import type { GameState } from '../replay';
	import type { MapTarget } from '../lesson';
	import { RIFT_SIZE, PITS } from '../rift';
	import { championName, initials, portraitUrl } from '../champions';
	import { clock } from '../replay';
	import type { Visibility } from '../vision';

	export let replay: GameReplay;
	export let state: GameState;
	/** The side the lesson is told from; drawn slightly larger. */
	export let team: Side;
	/** Map decision targets (pretest + comparison). */
	export let targets: MapTarget[] = [];
	export let pickable = false;
	export let picked: string | null = null;
	export let actualTargetId: string | null = null;
	export let targetNotes: Record<string, string> = {};
	/** Movement arrows, e.g. where each champion went over the next minute. */
	export let arrows: {
		id: number;
		from: { x: number; y: number };
		to: { x: number; y: number };
	}[] = [];
	/** Events to mark on the map (kills, objectives, towers). */
	export let marks: GameEvent[] = [];
	/** Fog of war from `team`'s view; omit to show everyone. */
	export let visibility: Map<number, Visibility> | null = null;
	export let onPick: (id: string) => void = () => {};

	const S = RIFT_SIZE;
	const fy = (y: number) => S - y;
	const R = 520;
	const uid = Math.random().toString(36).slice(2, 8);

	const teamColor = (side: Side) => (side === 'blue' ? 'var(--team-blue)' : 'var(--team-red)');

	function killerSide(e: GameEvent): Side | null {
		if (e.type === 'kill') return replay.participants.find((p) => p.id === e.killer)?.team ?? null;
		if (e.type === 'objective') return e.team;
		return e.owner === 'blue' ? 'red' : 'blue';
	}

	// Portraits load from Data Dragon. Only draw one once it has loaded, so a
	// blocked or missing image leaves the initials instead of a broken icon.
	let loaded: Record<string, boolean> = {};
	onMount(() => {
		for (const p of replay.participants) {
			const img = new Image();
			img.onload = () => (loaded = { ...loaded, [p.champion]: true });
			img.src = portraitUrl(p.champion, replay.patch);
		}
	});

	function pick(id: string) {
		if (pickable) onPick(id);
	}
</script>

<svg viewBox="0 0 {S} {S}" class="rift" role="img" aria-label="Minimap at {clock(state.t)}">
	<defs>
		{#each state.champions as c (c.participant.id)}
			<clipPath id="clip-{uid}-{c.participant.id}">
				<circle cx={c.x} cy={fy(c.y)} r={R - 60} />
			</clipPath>
		{/each}
		<marker
			id="arrow-{uid}-blue"
			viewBox="0 0 10 10"
			refX="8"
			refY="5"
			markerWidth="4"
			markerHeight="4"
			orient="auto-start-reverse"
		>
			<path d="M0,0 L10,5 L0,10 z" fill="var(--team-blue)" />
		</marker>
		<marker
			id="arrow-{uid}-red"
			viewBox="0 0 10 10"
			refX="8"
			refY="5"
			markerWidth="4"
			markerHeight="4"
			orient="auto-start-reverse"
		>
			<path d="M0,0 L10,5 L0,10 z" fill="var(--team-red)" />
		</marker>
	</defs>

	<!-- terrain -->
	<rect width={S} height={S} rx="300" class="ground" />
	<circle cx="0" cy={S} r="4300" class="base base-blue" />
	<circle cx={S} cy="0" r="4300" class="base base-red" />
	<path class="river" d="M 1700 {fy(13300)} Q 4700 {fy(9900)} 7500 {fy(7500)} T 13300 {fy(1700)}" />
	<g class="lanes">
		<path
			d="M 1000 {fy(4300)} L 1000 {fy(12700)} Q 1000 {fy(14000)} 2300 {fy(14000)} L 10700 {fy(
				14000
			)}"
		/>
		<path d="M 3300 {fy(3300)} L 11700 {fy(11700)}" />
		<path
			d="M 4300 {fy(1000)} L 12700 {fy(1000)} Q 14000 {fy(1000)} 14000 {fy(2300)} L 14000 {fy(
				10700
			)}"
		/>
	</g>
	<circle cx={PITS.baron.x} cy={fy(PITS.baron.y)} r="650" class="pit" />
	<circle cx={PITS.dragon.x} cy={fy(PITS.dragon.y)} r="650" class="pit" />

	<!-- towers -->
	{#each state.towers as tw (tw.id)}
		<g class="tower" class:down={!tw.alive} style="--c:{teamColor(tw.team)}">
			<rect x={tw.x - 230} y={fy(tw.y) - 230} width="460" height="460" rx="80" />
			{#if !tw.alive}
				<path
					d="M {tw.x - 170} {fy(tw.y) - 170} L {tw.x + 170} {fy(tw.y) + 170} M {tw.x + 170} {fy(
						tw.y
					) - 170} L {tw.x - 170} {fy(tw.y) + 170}"
				/>
			{/if}
		</g>
	{/each}

	<!-- targets -->
	{#each targets as tg (tg.id)}
		<!-- Map targets are a pointer shortcut; the chip buttons beside the map are the accessible control. -->
		<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
		<g
			class="target"
			class:pickable
			class:picked={picked === tg.id}
			class:actual={actualTargetId === tg.id}
			on:click={() => pick(tg.id)}
		>
			(e.key === 'Enter' || e.key === ' ') && pick(tg.id)} >
			<circle cx={tg.x} cy={fy(tg.y)} r="1000" />
			<text x={tg.x} y={fy(tg.y) + 1450} class="target-label">{tg.label}</text>
			{#if targetNotes[tg.id]}
				<text x={tg.x} y={fy(tg.y) + 1900} class="target-note">{targetNotes[tg.id]}</text>
			{/if}
		</g>
	{/each}

	<!-- movement -->
	{#each arrows as a (a.id)}
		{@const side = replay.participants.find((p) => p.id === a.id)?.team ?? 'blue'}
		{#if Math.hypot(a.to.x - a.from.x, a.to.y - a.from.y) > 900}
			<line
				x1={a.from.x}
				y1={fy(a.from.y)}
				x2={a.to.x}
				y2={fy(a.to.y)}
				class="move"
				style="--c:{teamColor(side)}"
				marker-end="url(#arrow-{uid}-{side})"
			/>
		{/if}
	{/each}

	<!-- event marks -->
	{#each marks as m, i (i)}
		{@const side = killerSide(m)}
		<g class="mark" style="--c:{side ? teamColor(side) : '#aaa'}">
			{#if m.type === 'kill'}
				<path
					d="M {m.x - 260} {fy(m.y) - 260} L {m.x + 260} {fy(m.y) + 260} M {m.x + 260} {fy(m.y) -
						260} L {m.x - 260} {fy(m.y) + 260}"
				/>
			{:else}
				<rect
					x={m.x - 300}
					y={fy(m.y) - 300}
					width="600"
					height="600"
					transform="rotate(45 {m.x} {fy(m.y)})"
				/>
			{/if}
		</g>
	{/each}

	<!-- enemies under fog: ghost at last-seen spot -->
	{#each state.champions as c (c.participant.id)}
		{@const v = visibility?.get(c.participant.id)}
		{#if v && v.kind === 'last-seen'}
			<g class="ghost" style="--c:{teamColor(c.participant.team)}">
				<title>{championName(c.participant.champion)} · last seen {Math.round(v.agoS)}s ago</title>
				<circle cx={v.x} cy={fy(v.y)} r={R} />
				<text x={v.x} y={fy(v.y) + 150} class="initials">{initials(c.participant.champion)}</text>
				<text x={v.x} y={fy(v.y) + R + 420} class="ago">{Math.round(v.agoS)}s ago</text>
			</g>
		{/if}
	{/each}

	<!-- champions -->
	{#each state.champions.filter((c) => !visibility || visibility.get(c.participant.id)?.kind === 'visible') as c (c.participant.id)}
		{@const side = c.participant.team}
		{@const cy = fy(c.y)}
		<g
			class="champ"
			class:dead={c.deadFor > 0}
			class:ours={side === team}
			class:approx={!c.exact}
			style="--c:{teamColor(side)}"
		>
			<title>
				{championName(c.participant.champion)} · level {c.level} · {Math.round(c.hp)}/{c.hpMax} HP{c.deadFor >
				0
					? ` · dead, back in ≈${Math.ceil(c.deadFor)}s`
					: ''}{c.exact ? '' : ' · position approximate'}
			</title>
			<circle cx={c.x} {cy} r={R} class="ring" />
			<text x={c.x} y={cy + 150} class="initials">{initials(c.participant.champion)}</text>
			{#if loaded[c.participant.champion]}
				<image
					href={portraitUrl(c.participant.champion, replay.patch)}
					x={c.x - R}
					y={cy - R}
					width={R * 2}
					height={R * 2}
					clip-path="url(#clip-{uid}-{c.participant.id})"
					preserveAspectRatio="xMidYMid slice"
				/>
			{/if}
			{#if c.deadFor > 0}
				<text x={c.x} y={cy + R + 420} class="timer">{Math.ceil(c.deadFor)}s</text>
			{:else}
				<g class="bars" class:stale={!c.vitalsExact}>
					<rect x={c.x - 650} y={cy - R - 420} width="1300" height="190" class="bar-bg" />
					<rect
						x={c.x - 650}
						y={cy - R - 420}
						width={1300 * Math.max(0, Math.min(1, c.hp / c.hpMax))}
						height="190"
						class="hp"
					/>
					{#if c.mpMax > 0}
						<rect x={c.x - 650} y={cy - R - 215} width="1300" height="100" class="bar-bg" />
						<rect
							x={c.x - 650}
							y={cy - R - 215}
							width={1300 * Math.max(0, Math.min(1, c.mp / c.mpMax))}
							height="100"
							class="mp"
						/>
					{/if}
				</g>
			{/if}
		</g>
	{/each}
</svg>

<style>
	.rift {
		width: 100%;
		height: auto;
		display: block;
		border-radius: 10px;
		--team-blue: #4ea1ff;
		--team-red: #ff5a5f;
	}
	.ground {
		fill: #13211b;
	}
	.base-blue {
		fill: color-mix(in srgb, #4ea1ff 14%, transparent);
	}
	.base-red {
		fill: color-mix(in srgb, #ff5a5f 14%, transparent);
	}
	.river {
		fill: none;
		stroke: #1b3a4f;
		stroke-width: 1500;
		stroke-linecap: round;
	}
	.lanes path {
		fill: none;
		stroke: #3a3a2c;
		stroke-width: 850;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.pit {
		fill: #12303f;
		stroke: #2c5a72;
		stroke-width: 120;
	}
	.tower rect {
		fill: var(--c);
		stroke: #0b0f0d;
		stroke-width: 90;
	}
	.tower.down rect {
		fill: #1b231f;
		stroke: var(--c);
		stroke-width: 70;
		opacity: 0.6;
	}
	.tower.down path {
		stroke: var(--c);
		stroke-width: 80;
		opacity: 0.7;
	}
	.target circle {
		fill: rgba(255, 255, 255, 0.04);
		stroke: rgba(255, 255, 255, 0.55);
		stroke-width: 70;
		stroke-dasharray: 220 160;
	}
	.target.pickable {
		cursor: pointer;
	}
	.target.pickable:hover circle,
	.target.pickable:active circle {
		fill: rgba(255, 255, 255, 0.12);
		stroke: #fff;
	}
	.target.picked circle {
		fill: color-mix(in srgb, var(--accent) 30%, transparent);
		stroke: var(--accent);
		stroke-dasharray: none;
	}
	.target.actual circle {
		stroke: var(--gold);
		stroke-dasharray: none;
	}
	.target-label,
	.target-note {
		text-anchor: middle;
		fill: #e8e8e8;
		font-size: 430px;
		font-weight: 600;
		paint-order: stroke;
		stroke: #0b0f0d;
		stroke-width: 120;
	}
	.target-note {
		font-size: 380px;
		font-weight: 500;
		fill: #c9c9c9;
	}
	.move {
		stroke: var(--c);
		stroke-width: 130;
		stroke-dasharray: 300 200;
		opacity: 0.85;
	}
	.mark path {
		stroke: var(--c);
		stroke-width: 150;
		stroke-linecap: round;
	}
	.mark rect {
		fill: var(--c);
		stroke: #0b0f0d;
		stroke-width: 80;
	}
	.champ .ring {
		fill: #1d2430;
		stroke: var(--c);
		stroke-width: 120;
	}
	.champ.ours .ring {
		stroke-width: 170;
	}
	.champ .initials {
		text-anchor: middle;
		font-size: 420px;
		font-weight: 700;
		fill: #e8e8e8;
	}
	.ghost circle {
		fill: none;
		stroke: var(--c);
		stroke-width: 90;
		stroke-dasharray: 120 140;
		opacity: 0.7;
	}
	.ghost .initials {
		text-anchor: middle;
		font-size: 380px;
		font-weight: 700;
		fill: var(--c);
		opacity: 0.7;
	}
	.ghost .ago {
		text-anchor: middle;
		font-size: 340px;
		fill: #cfcfcf;
		paint-order: stroke;
		stroke: #000;
		stroke-width: 100;
	}
	.champ.approx {
		opacity: 0.6;
	}
	.champ.approx .ring {
		stroke-dasharray: 160 110;
	}
	.bar-bg {
		fill: #0b0f0d;
		stroke: #0b0f0d;
		stroke-width: 50;
	}
	.hp {
		fill: var(--c);
	}
	.mp {
		fill: #7dd3fc;
	}
	.bars.stale {
		opacity: 0.45;
	}
	.champ.dead {
		opacity: 0.45;
		filter: grayscale(1);
	}
	.champ .timer {
		text-anchor: middle;
		font-size: 400px;
		font-weight: 700;
		fill: #fff;
		paint-order: stroke;
		stroke: #000;
		stroke-width: 110;
	}
</style>
