<script lang="ts">
	import Minimap from '$lib/game/components/Minimap.svelte';
	import GoldSwing from '$lib/game/components/GoldSwing.svelte';
	import InterventionPrompt from '$lib/lesson/components/InterventionPrompt.svelte';
	import IntentReceipt from '$lib/lesson/components/IntentReceipt.svelte';
	import StreakBadge from '$lib/lesson/components/StreakBadge.svelte';
	import { streakStore } from '$lib/lesson/stores/streakStore';
	import { onboardingStore } from '$lib/lesson/stores/onboardingStore';
	import { stateAt, eventsBetween, goldDiffSeries, clock, participantById } from '$lib/game/replay';
	import { factsAt } from '$lib/game/facts';
	import { championName } from '$lib/game/champions';
	import type { GameEvent } from '$lib/game/types';
	import type { PageData } from './$types';

	export let data: PageData;
	$: ({ lesson, replay } = data);

	type Phase = 'pause' | 'reveal' | 'intent' | 'committed';
	let phase: Phase = 'pause';
	let picked: string | null = null;
	let intent: string | null = null;
	let cursor = 0; // index into beats; -1 = paused moment

	$: enemy = lesson.team === 'blue' ? 'red' : 'blue';
	$: paused = stateAt(replay, lesson.t);
	$: facts = factsAt(paused, lesson.team, lesson.targets);
	$: beats = eventsBetween(replay, lesson.t, lesson.t + lesson.playForwardS);
	$: viewT = phase === 'pause' || cursor < 0 ? lesson.t : (beats[cursor]?.t ?? lesson.t);
	$: view = stateAt(replay, viewT);
	$: arrows =
		phase === 'pause'
			? []
			: paused.champions
					.filter((c) => c.participant.team === lesson.team)
					.map((c) => {
						const now = view.champions.find((v) => v.participant.id === c.participant.id)!;
						return { id: c.participant.id, from: c, to: now };
					});
	$: marks = phase === 'pause' ? [] : beats.filter((e) => e.t <= viewT);
	$: series = goldDiffSeries(replay, lesson.team);
	$: endT = lesson.t + lesson.playForwardS;
	$: kills = (side: string, t: number) =>
		replay.events.filter(
			(e) => e.type === 'kill' && e.t <= t && participantById(replay, e.killer)?.team === side
		).length;
	$: notes =
		phase === 'pause' || !lesson.comparison
			? {}
			: Object.fromEntries(
					lesson.comparison.byTarget.map((b) => [b.targetId, `${b.won} of ${b.games} won`])
				);
	$: maxGames = lesson.comparison ? Math.max(...lesson.comparison.byTarget.map((b) => b.games)) : 1;

	function name(id: number) {
		const p = participantById(replay, id);
		return p ? championName(p.champion) : 'Minions';
	}
	function sideOf(id: number) {
		return participantById(replay, id)?.team ?? null;
	}
	function beatText(e: GameEvent): string {
		if (e.type === 'kill') return `${name(e.killer)} kills ${name(e.victim)}`;
		if (e.type === 'objective') {
			const label =
				e.monster === 'RIFTHERALD'
					? 'Rift Herald'
					: e.monster === 'DRAGON'
						? 'Dragon'
						: e.monster === 'BARON_NASHOR'
							? 'Baron'
							: 'Voidgrubs';
			return `${name(e.killer)} takes ${label}`;
		}
		const whose = e.owner === lesson.team ? 'Your' : 'Their';
		const lane = e.lane.replace('_LANE', '').toLowerCase();
		const tier = (e.tower ?? '').startsWith('INNER')
			? ' inner'
			: (e.tower ?? '').startsWith('BASE')
				? ' base'
				: '';
		return `${whose} ${lane}${tier} tower falls`;
	}
	function beatSide(e: GameEvent) {
		if (e.type === 'kill') return sideOf(e.killer);
		if (e.type === 'objective') return e.team;
		return e.owner === 'blue' ? 'red' : 'blue';
	}

	function reveal() {
		phase = 'reveal';
		cursor = -1;
	}
	let timer: ReturnType<typeof setInterval> | null = null;
	function play() {
		stop();
		cursor = -1;
		timer = setInterval(() => {
			if (cursor >= beats.length - 1) return stop();
			cursor += 1;
		}, 900);
	}
	function stop() {
		if (timer) clearInterval(timer);
		timer = null;
	}
	function commit(text: string) {
		intent = text;
		phase = 'committed';
		streakStore.recordIntervention(text);
		onboardingStore.recordLessonCompleted();
	}
</script>

<svelte:head>
	<title>Game review — howtowin.lol</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<div class="page">
	<header class="head">
		<div>
			<a class="back" href="/games">← games</a>
			<h1>
				<span class="clock">{clock(viewT)}</span>
				<span class="side side-{lesson.team}">You: {lesson.team} side</span>
				<span class="result" class:won={replay.winner === lesson.team}>
					{replay.winner === lesson.team ? 'win' : 'loss'}
				</span>
			</h1>
			<p class="score">
				Kills {kills(lesson.team, viewT)}–{kills(enemy, viewT)} ·
				{view.goldDiff[lesson.team] >= 0 ? 'up' : 'down'}
				{Math.abs(Math.round(view.goldDiff[lesson.team] / 10) * 10).toLocaleString()} gold
			</p>
		</div>
		<StreakBadge />
	</header>

	<div class="grid">
		<div class="map-col">
			<Minimap
				{replay}
				state={view}
				team={lesson.team}
				targets={lesson.targets}
				pickable={phase === 'pause'}
				{picked}
				actualTargetId={phase === 'pause' ? null : lesson.actualTargetId}
				targetNotes={notes}
				{arrows}
				{marks}
				onPick={(id) => (picked = id)}
			/>
			{#if phase !== 'pause'}
				<div class="scrub">
					<button type="button" class="ghost" on:click={timer ? stop : play}>
						{timer ? 'Pause' : '▶ Play it forward'}
					</button>
					<input
						type="range"
						min="-1"
						max={beats.length - 1}
						bind:value={cursor}
						on:input={stop}
						aria-label="Step through what happened"
					/>
				</div>
			{/if}
			<p class="caveat">
				Positions come from Riot's match timeline, sampled once a minute and pinned to exact spots
				at kills. Timers are approximate.
			</p>
		</div>

		<aside class="panel">
			{#if phase === 'pause'}
				<section>
					<h2>Paused at {clock(lesson.t)}</h2>
					<ul class="facts">
						{#each facts as f}
							<li class="fact fact-{f.kind}" class:theirs={f.side && f.side !== lesson.team}>
								{f.text}
							</li>
						{/each}
					</ul>
				</section>
				<section class="question">
					<p class="q">{lesson.question}</p>
					<p class="hint">Tap a spot on the map.</p>
					<div class="chips">
						{#each lesson.targets as tg (tg.id)}
							<button
								type="button"
								class="chip"
								class:on={picked === tg.id}
								on:click={() => (picked = tg.id)}>{tg.label}</button
							>
						{/each}
					</div>
					<button type="button" class="primary" disabled={!picked} on:click={reveal}>
						Show what happened
					</button>
				</section>
			{:else if phase === 'reveal'}
				<section>
					<h2>What your team did</h2>
					<p>{lesson.actualSummary}</p>
				</section>
				<section>
					<h2>Then</h2>
					<ol class="beats">
						{#each beats as e, i}
							<li>
								<button
									type="button"
									class="beat"
									class:on={i === cursor}
									class:past={i < cursor}
									on:click={() => {
										stop();
										cursor = i;
									}}
								>
									<span class="t">{clock(e.t)}</span>
									<span class="dot dot-{beatSide(e)}"></span>
									<span>{beatText(e)}</span>
								</button>
							</li>
						{/each}
					</ol>
				</section>
				<GoldSwing {series} from={lesson.t} to={endT} />
				{#if lesson.comparison}
					<section>
						<h2>
							From similar spots
							{#if lesson.illustrative}<span class="tag">illustrative numbers</span>{/if}
						</h2>
						<p class="sub">
							Where teams went in {lesson.comparison.games} games that reached a state like this one.
						</p>
						<ul class="bars">
							{#each lesson.comparison.byTarget as b (b.targetId)}
								{@const tg = lesson.targets.find((x) => x.id === b.targetId)}
								<li class:mine={picked === b.targetId}>
									<span class="label">
										{tg?.label}{picked === b.targetId ? ' · your pick' : ''}
									</span>
									<span class="bar" style="width:{(b.games / maxGames) * 100}%">
										<span class="won" style="width:{(b.won / b.games) * 100}%"></span>
									</span>
									<span class="n">{b.won} of {b.games} won</span>
								</li>
							{/each}
						</ul>
					</section>
				{/if}
				<button type="button" class="primary" on:click={() => (phase = 'intent')}>
					Set one thing for next game →
				</button>
			{:else if phase === 'intent'}
				<InterventionPrompt onAccept={commit} />
			{:else if intent}
				<IntentReceipt {intent} />
			{/if}
		</aside>
	</div>

	{#if lesson.illustrative}
		<p class="sample-note">
			Real ranked game ({replay.matchId}, patch {replay.patch.split('.').slice(0, 2).join('.')}).
			The question and the similar-spots numbers are hand-written until the model produces them.
		</p>
	{/if}
</div>

<style>
	.page {
		max-width: 1080px;
		display: flex;
		flex-direction: column;
		gap: 18px;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 12px;
	}
	.back {
		font-size: 0.85rem;
		color: var(--text-muted);
	}
	h1 {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 10px;
		font-size: 1rem;
		font-weight: 500;
		margin-top: 4px;
	}
	.clock {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 1.8rem;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	.side,
	.result,
	.tag {
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		padding: 2px 8px;
		border-radius: 999px;
		border: 1px solid var(--border);
	}
	.side-blue {
		color: #4ea1ff;
	}
	.side-red {
		color: #ff5a5f;
	}
	.result {
		color: var(--loss);
	}
	.result.won {
		color: var(--win);
	}
	.score {
		color: var(--text-muted);
		font-size: 0.85rem;
		font-variant-numeric: tabular-nums;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
		gap: 24px;
		align-items: start;
	}
	@media (max-width: 860px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
	.map-col {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.caveat {
		font-size: 0.72rem;
		color: var(--text-muted);
	}
	.scrub {
		display: flex;
		gap: 12px;
		align-items: center;
	}
	.scrub input {
		flex: 1;
		accent-color: var(--gold);
	}
	.panel {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}
	h2 {
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-muted);
		font-weight: 600;
		margin-bottom: 8px;
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.tag {
		color: var(--gold);
		text-transform: none;
		letter-spacing: 0;
	}
	.facts {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.fact {
		font-size: 0.92rem;
		padding-left: 14px;
		position: relative;
	}
	.fact::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0.55em;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--text-muted);
	}
	.fact-down.theirs::before,
	.fact-base.theirs::before {
		background: var(--win);
	}
	.fact-down:not(.theirs)::before,
	.fact-base:not(.theirs)::before {
		background: var(--loss);
	}
	.fact-objective::before {
		background: var(--gold);
	}
	.fact-race::before {
		background: #4ea1ff;
	}
	.question {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding-top: 16px;
		border-top: 1px solid var(--border);
	}
	.q {
		font-size: 1.1rem;
		font-weight: 600;
		line-height: 1.4;
	}
	.hint,
	.sub {
		font-size: 0.82rem;
		color: var(--text-muted);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.chip {
		background: var(--bg-surface);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 6px 12px;
		font: inherit;
		font-size: 0.85rem;
		cursor: pointer;
	}
	.chip.on {
		border-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 20%, var(--bg-surface));
	}
	.primary {
		align-self: flex-start;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 10px 16px;
		font: inherit;
		cursor: pointer;
	}
	.primary:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.ghost {
		background: none;
		border: 1px solid var(--border);
		color: var(--text);
		border-radius: 6px;
		padding: 6px 12px;
		font: inherit;
		font-size: 0.85rem;
		cursor: pointer;
		white-space: nowrap;
	}
	.beats {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.beat {
		width: 100%;
		display: flex;
		gap: 10px;
		align-items: center;
		background: none;
		border: none;
		color: var(--text-muted);
		font: inherit;
		font-size: 0.88rem;
		padding: 5px 8px;
		border-radius: 6px;
		cursor: pointer;
		text-align: left;
	}
	.beat.past {
		color: var(--text);
	}
	.beat.on {
		color: var(--text);
		background: var(--bg-surface);
	}
	.beat .t {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-variant-numeric: tabular-nums;
		width: 3.2em;
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}
	.dot-blue {
		background: #4ea1ff;
	}
	.dot-red {
		background: #ff5a5f;
	}
	.bars {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.bars li {
		display: grid;
		grid-template-columns: 11em 1fr 6.5em;
		gap: 10px;
		align-items: center;
		font-size: 0.85rem;
	}
	.bars li.mine .label {
		color: var(--accent);
	}
	.bar {
		height: 10px;
		background: color-mix(in srgb, var(--loss) 55%, transparent);
		border-radius: 3px;
		overflow: hidden;
		display: block;
	}
	.won {
		display: block;
		height: 100%;
		background: var(--win);
	}
	.n {
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	.sample-note {
		font-size: 0.78rem;
		color: var(--text-muted);
		border-top: 1px solid var(--border);
		padding-top: 12px;
	}
</style>
