<script lang="ts">
	import { onMount } from 'svelte';
	import Minimap from '$lib/game/components/Minimap.svelte';
	import GoldSwing from '$lib/game/components/GoldSwing.svelte';
	import InterventionPrompt from '$lib/lesson/components/InterventionPrompt.svelte';
	import IntentReceipt from '$lib/lesson/components/IntentReceipt.svelte';
	import StreakBadge from '$lib/lesson/components/StreakBadge.svelte';
	import { streakStore } from '$lib/lesson/stores/streakStore';
	import { onboardingStore } from '$lib/lesson/stores/onboardingStore';
	import { stateAt, eventsBetween, goldDiffSeries, clock, participantById } from '$lib/game/replay';
	import { factsAt } from '$lib/game/facts';
	import { visibilityAt } from '$lib/game/vision';
	import { championName } from '$lib/game/champions';
	import type { GameEvent } from '$lib/game/types';
	import type { PageData } from './$types';

	export let data: PageData;
	$: ({ lesson, replay } = data);

	type Phase = 'intro' | 'puzzle' | 'reveal' | 'intent' | 'committed';
	const LEAD_IN_S = 60;
	let phase: Phase = 'intro';
	let picked: string | null = null;
	let intent: string | null = null;
	let now = data.lesson.t - LEAD_IN_S;
	let playing = false;
	let speed = 8; // game seconds per real second

	$: enemy = lesson.team === 'blue' ? 'red' : 'blue';
	$: startT = lesson.t - LEAD_IN_S;
	$: endT = lesson.t + lesson.playForwardS;
	$: paused = stateAt(replay, lesson.t);
	$: pausedVis = visibilityAt(replay, lesson.t, lesson.team, paused);
	$: facts = factsAt(paused, lesson.team, lesson.targets, pausedVis);
	let showAll = true;

	// "Where is X?" step: before the decision, ask the player to place the
	// enemy they couldn't see (jungler first, else the longest-unseen one).
	type Track = 'ask' | 'answered' | 'done';
	$: unseenTarget = (() => {
		const hidden = replay.participants.filter(
			(p) => p.team !== lesson.team && pausedVis.get(p.id)?.kind !== 'visible'
		);
		return hidden.find((p) => p.role === 'JUNGLE') ?? hidden[0] ?? null;
	})();
	let track: Track = 'ask';
	let guess: { x: number; y: number } | null = null;
	$: trackActive = phase === 'puzzle' && unseenTarget !== null && track !== 'done';
	$: actual = unseenTarget
		? (paused.champions.find((c) => c.participant.id === unseenTarget.id) ?? null)
		: null;
	$: missBy = guess && actual ? Math.hypot(guess.x - actual.x, guess.y - actual.y) : 0;
	$: unseenInfo = unseenTarget ? pausedVis.get(unseenTarget.id) : undefined;
	function placeGuess(x: number, y: number) {
		if (track !== 'ask') return;
		guess = { x, y };
		track = 'answered';
	}
	// Fog applies while the player is "in the game" (lead-in and puzzle);
	// after answering, the full board is revealed unless they toggle it back.
	$: fogged = phase === 'intro' || phase === 'puzzle' || !showAll;
	$: second = Math.floor(now);
	$: baseVis =
		fogged && phase !== 'intent' && phase !== 'committed'
			? visibilityAt(replay, second, lesson.team)
			: null;
	$: vis =
		baseVis && unseenTarget && phase === 'puzzle' && track !== 'ask'
			? new Map([...baseVis, [unseenTarget.id, { kind: 'visible' as const }]])
			: baseVis;
	$: beats = eventsBetween(replay, lesson.t, endT);
	$: view = stateAt(replay, now);
	$: marks = replay.events.filter((e) => e.t > startT && e.t <= now && e.type !== 'building');
	$: series = goldDiffSeries(replay, lesson.team);
	$: kills = (side: string, t: number) =>
		replay.events.filter(
			(e) => e.type === 'kill' && e.t <= t && participantById(replay, e.killer)?.team === side
		).length;
	$: notes =
		phase === 'intro' || phase === 'puzzle' || !lesson.comparison
			? {}
			: Object.fromEntries(
					lesson.comparison.byTarget.map((b) => [b.targetId, `${b.won} of ${b.games} won`])
				);
	$: maxGames = lesson.comparison ? Math.max(...lesson.comparison.byTarget.map((b) => b.games)) : 1;
	$: onMinute = Math.abs(now - Math.round(now / 60) * 60) <= 2;
	$: currentBeat = beats.reduce((acc, e, i) => (e.t <= now ? i : acc), -1);

	// Playback: rAF loop advancing `now`; stops at the puzzle moment during the
	// intro and at the end of the window after the reveal.
	let raf = 0;
	let last = 0;
	function tick(ts: number) {
		if (!playing) return;
		const dt = last ? (ts - last) / 1000 : 0;
		last = ts;
		const stopAt = phase === 'intro' ? lesson.t : endT;
		now = Math.min(stopAt, now + dt * speed);
		if (now >= stopAt) {
			playing = false;
			if (phase === 'intro') phase = 'puzzle';
			return;
		}
		raf = requestAnimationFrame(tick);
	}
	function play() {
		if (phase === 'reveal' && now >= endT) now = lesson.t;
		playing = true;
		last = 0;
		raf = requestAnimationFrame(tick);
	}
	function pause() {
		playing = false;
		cancelAnimationFrame(raf);
	}
	function skipToPuzzle() {
		pause();
		now = lesson.t;
		phase = 'puzzle';
	}
	function seek(t: number) {
		pause();
		now = t;
	}
	onMount(() => {
		play();
		return pause;
	});

	function reveal() {
		phase = 'reveal';
		now = lesson.t;
		play();
	}
	function commit(text: string) {
		intent = text;
		phase = 'committed';
		streakStore.recordIntervention(text);
		onboardingStore.recordLessonCompleted();
	}

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
				<span class="clock">{clock(now)}</span>
				<span class="side side-{lesson.team}">You: {lesson.team} side</span>
				<span class="result" class:won={replay.winner === lesson.team}>
					{replay.winner === lesson.team ? 'win' : 'loss'}
				</span>
			</h1>
			<p class="score">
				Kills {kills(lesson.team, now)}–{kills(enemy, now)} ·
				{view.goldDiff[lesson.team] >= 0 ? 'up' : 'down'}
				{Math.abs(Math.round(view.goldDiff[lesson.team] / 10) * 10).toLocaleString()} gold
			</p>
		</div>
		<StreakBadge />
	</header>

	<div class="grid">
		<div class="map-col">
			<div class="map-wrap">
				<Minimap
					{replay}
					state={view}
					team={lesson.team}
					targets={phase === 'intro' || trackActive ? [] : lesson.targets}
					pickable={phase === 'puzzle' && !trackActive}
					onMapClick={trackActive && track === 'ask' ? placeGuess : null}
					guess={phase === 'puzzle' && track !== 'done' ? guess : null}
					{picked}
					actualTargetId={phase === 'reveal' ? lesson.actualTargetId : null}
					targetNotes={notes}
					{marks}
					visibility={phase === 'reveal' && showAll ? null : vis}
					onPick={(id) => (picked = id)}
				/>
				{#if phase === 'puzzle'}
					<div class="stamp">Paused · {clock(lesson.t)}</div>
				{/if}
			</div>
			{#if phase !== 'puzzle'}
				<div class="scrub">
					<button type="button" class="ghost" on:click={playing ? pause : play}>
						{playing ? 'Pause' : '▶ Play'}
					</button>
					<input
						type="range"
						min={phase === 'intro' ? startT : lesson.t}
						max={phase === 'intro' ? lesson.t : endT}
						step="1"
						value={now}
						on:input={(e) => seek(+e.currentTarget.value)}
						aria-label="Replay time"
					/>
					{#if phase === 'reveal'}
						<label class="toggle"
							><input type="checkbox" bind:checked={showAll} /> Show everyone</label
						>
					{/if}
					<select bind:value={speed} aria-label="Playback speed">
						<option value={4}>4×</option>
						<option value={8}>8×</option>
						<option value={16}>16×</option>
					</select>
				</div>
			{/if}
			<p class="legend">
				<span class="solid">Solid</span> = recorded by Riot at this moment (every minute mark, every
				kill). <span class="faded">Faded</span> = in between: movement is a straight-line guess and
				HP shows the last recorded value. Before you answer, enemies only appear when your champions
				or towers could see them; dashed ghosts mark where they were last seen. Wards aren't
				recorded, so you may have seen more than this.
				<b class:live={onMinute}
					>{onMinute ? 'On a minute mark: everything shown is recorded.' : ''}</b
				>
			</p>
		</div>

		<aside class="panel">
			{#if phase === 'intro'}
				<section>
					<h2>Watch the minute before</h2>
					<p>
						The replay plays {clock(startT)} to {clock(lesson.t)}, then stops at the moment that
						mattered and asks what you'd do.
					</p>
					<button type="button" class="link" on:click={skipToPuzzle}>Skip to the puzzle →</button>
				</section>
			{:else if trackActive && unseenTarget}
				<section class="question">
					<h2>Before you decide</h2>
					<p class="q">
						{championName(unseenTarget.champion)}
						{unseenInfo?.kind === 'last-seen'
							? `was last seen ${Math.round(unseenInfo.agoS)}s ago.`
							: "hasn't been seen for two minutes."}
						Where are they now?
					</p>
					{#if track === 'ask'}
						<p class="hint">Tap the map where you think they are.</p>
					{:else}
						<p class="verdict" class:close={missBy < 2000}>
							{missBy < 2000
								? 'Close.'
								: missBy < 4500
									? 'Right side of the map, wrong spot.'
									: 'Not there.'}
							{championName(unseenTarget.champion)} was about {Math.round(missBy / 380)}s' walk from
							your guess.
						</p>
						<p class="sub">
							The solid icon is where they really were at {clock(lesson.t)}. Tracking who you can't
							see is half of the next decision.
						</p>
						<button type="button" class="primary" on:click={() => (track = 'done')}>
							Next: the decision →
						</button>
					{/if}
				</section>
			{:else if phase === 'puzzle'}
				<section>
					<h2>The board at {clock(lesson.t)}</h2>
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
						Lock it in and play on
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
									class:on={i === currentBeat}
									class:past={i < currentBeat}
									on:click={() => seek(e.t)}
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
	.map-wrap {
		position: relative;
	}
	.stamp {
		position: absolute;
		top: 12px;
		left: 50%;
		transform: translateX(-50%);
		background: rgba(0, 0, 0, 0.7);
		border: 1px solid var(--gold);
		color: var(--gold);
		font-size: 0.8rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		padding: 4px 12px;
		border-radius: 999px;
	}
	.legend {
		font-size: 0.75rem;
		color: var(--text-muted);
		line-height: 1.5;
	}
	.toggle {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 0.8rem;
		color: var(--text-muted);
		white-space: nowrap;
	}
	.legend .solid {
		color: var(--text);
		font-weight: 600;
	}
	.legend .faded {
		opacity: 0.6;
		font-weight: 600;
	}
	.legend b {
		display: block;
		color: var(--win);
		font-weight: 500;
		min-height: 1.5em;
	}
	select {
		background: var(--bg-surface);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 5px 6px;
		font: inherit;
		font-size: 0.85rem;
	}
	.link {
		background: none;
		border: none;
		color: var(--accent);
		font: inherit;
		cursor: pointer;
		padding: 0;
		margin-top: 8px;
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
	.fact-fog::before {
		background: transparent;
		border: 1px dashed var(--text-muted);
	}
	.fact-race::before {
		background: #4ea1ff;
	}
	.verdict {
		font-weight: 600;
		color: var(--loss);
	}
	.verdict.close {
		color: var(--win);
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
