<script lang="ts">
	import { MODEL_ONLY_EXTRAS, toLessonViewModel } from '$lib/lesson/adapter';
	import LessonStage from '$lib/lesson/components/LessonStage.svelte';
	import StreakBadge from '$lib/lesson/components/StreakBadge.svelte';
	import type { PageData } from './$types';

	export let data: PageData;

	$: lesson = data.lesson;
	$: viewModel = toLessonViewModel(lesson, MODEL_ONLY_EXTRAS);
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
				<span class="result" class:won={lesson.player_won}>
					{lesson.player_won ? 'win' : 'loss'}
				</span>
			</h1>
		</div>
		<StreakBadge />
	</header>

	<p class="sample-note">
		Sample game. These lessons are mock model output while the UI is built separately from the
		model.
	</p>

	{#key sessionKey}
		<LessonStage {viewModel} fixtureId={sessionKey} />
	{/key}
	<footer class="provenance">
		model {viewModel.raw.checkpoint_sha} · index {viewModel.raw.index_checkpoint_sha} ·
		{viewModel.raw.n_mid_anchors} mid-game minutes checked
	</footer>
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
	.sample-note {
		font-size: 0.8rem;
		color: var(--gold);
		border: 1px solid color-mix(in srgb, var(--gold) 40%, transparent);
		border-radius: 6px;
		padding: 6px 10px;
	}
</style>
