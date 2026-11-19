<script lang="ts">
	import { MODEL_ONLY_EXTRAS, toLessonViewModel } from '$lib/lesson/adapter';
	import LessonStage from '$lib/lesson/components/LessonStage.svelte';
	import StreakBadge from '$lib/lesson/components/StreakBadge.svelte';
	import type { PageData } from './$types';

	export let data: PageData;

	$: lesson = data.result.ok ? data.result.lesson : null;
	$: viewModel = lesson ? toLessonViewModel(lesson, MODEL_ONLY_EXTRAS) : null;
	$: sessionKey = `${data.matchId}-${data.team}`;
</script>

<svelte:head>
	<title>Game review — howtowin.lol</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<div class="page">
	<header class="head">
		<div>
			<a class="back" href="/games">← another game</a>
			<h1>
				<span class="mono">{data.matchId}</span>
				<span class="side side-{data.team}">{data.team} side</span>
				{#if lesson}
					<span class="result" class:won={lesson.player_won}>
						{lesson.player_won ? 'win' : 'loss'}
					</span>
				{/if}
			</h1>
		</div>
		<StreakBadge />
	</header>

	{#if viewModel}
		{#key sessionKey}
			<LessonStage {viewModel} fixtureId={sessionKey} />
		{/key}
		<footer class="provenance">
			model {viewModel.raw.checkpoint_sha} · index {viewModel.raw.index_checkpoint_sha} ·
			{viewModel.raw.n_mid_anchors} mid-game minutes checked
		</footer>
	{:else if !data.result.ok}
		<section class="unavailable">
			{#if data.result.reason === 'engine_off'}
				<h2>Game review isn't running on this server yet.</h2>
				<p>
					Lessons come from the model, which runs on a separate machine for now. <a href="/review"
						>Join the waitlist</a
					> and we'll tell you when it's live.
				</p>
			{:else if data.result.reason === 'model_unavailable'}
				<h2>The model isn't loaded here.</h2>
				<p>This server has the lesson engine turned on, but no trained checkpoint or index.</p>
			{:else}
				<h2>We couldn't build a lesson for this game.</h2>
				<p>Some games aren't in our dataset yet, or the model hit an error.</p>
			{/if}
			<details>
				<summary>Details</summary>
				<pre>{data.result.detail}</pre>
			</details>
		</section>
	{/if}
</div>

<style>
	.page {
		max-width: 680px;
		display: flex;
		flex-direction: column;
		gap: 20px;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 12px;
	}
	.back {
		font-size: 0.9rem;
		color: var(--text-muted);
	}
	.back:hover {
		color: var(--accent);
	}
	h1 {
		font-size: 1.15rem;
		font-weight: 500;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 10px;
		margin-top: 6px;
	}
	.mono {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		overflow-wrap: anywhere;
	}
	.side,
	.result {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		padding: 2px 8px;
		border-radius: 999px;
		border: 1px solid var(--border);
		color: var(--text-muted);
	}
	.side-blue {
		color: #60a5fa;
	}
	.side-red {
		color: #f87171;
	}
	.result {
		color: var(--loss);
	}
	.result.won {
		color: var(--win);
	}
	.provenance {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		overflow-wrap: anywhere;
	}
	.unavailable {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 20px 18px;
		background: var(--bg-surface);
		border: 1px dashed var(--border);
		border-radius: 6px;
	}
	.unavailable h2 {
		font-size: 1.05rem;
	}
	.unavailable p {
		color: var(--text-muted);
		line-height: 1.6;
	}
	details {
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	pre {
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		margin-top: 6px;
	}
</style>
