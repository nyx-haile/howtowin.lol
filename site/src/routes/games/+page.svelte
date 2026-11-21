<script lang="ts">
	import { sampleGames } from '$lib/lesson/samples';
</script>

<svelte:head>
	<title>Review a game — howtowin.lol</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<section class="games">
	<h2>Review a game</h2>
	<p class="sub">
		Each lesson finds the mid-game minute where games like yours split between wins and losses.
		These are sample games while the model is built separately.
	</p>

	<ul>
		{#each sampleGames as g (g.matchId + g.team)}
			<li>
				<a href="/games/{g.matchId}-{g.team}">
					<span class="row">
						<span class="mono">{g.matchId}</span>
						<span class="side side-{g.team}">{g.team} side</span>
						<span class="result" class:won={g.lesson.player_won}>
							{g.lesson.player_won ? 'win' : 'loss'}
						</span>
					</span>
					<span class="summary">{g.summary}</span>
				</a>
			</li>
		{/each}
	</ul>
</section>

<style>
	.games {
		max-width: 680px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	h2 {
		font-size: 1.4rem;
	}
	.sub {
		color: var(--text-muted);
		line-height: 1.6;
	}
	ul {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	a {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text);
	}
	a:hover {
		border-color: var(--accent);
		color: var(--text);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 10px;
	}
	.mono {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
	}
	.side,
	.result {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		padding: 2px 8px;
		border-radius: 999px;
		border: 1px solid var(--border);
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
	.summary {
		font-size: 0.85rem;
		color: var(--text-muted);
		line-height: 1.5;
	}
</style>
