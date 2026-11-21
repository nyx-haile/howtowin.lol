<script lang="ts">
	import { sampleLessons, sampleReplays } from '$lib/game/samples';
	import { clock } from '$lib/game/replay';
	import { championName } from '$lib/game/champions';
</script>

<svelte:head>
	<title>Review a game — howtowin.lol</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<section class="games">
	<h2>Review a game</h2>
	<p class="sub">Each review pauses your game at one moment and asks what you'd do from there.</p>
	<ul>
		{#each sampleLessons as l (l.matchId + l.team)}
			{@const r = sampleReplays[l.matchId]}
			<li>
				<a href="/games/{l.matchId}-{l.team}">
					<span class="row">
						<span class="t">{clock(l.t)}</span>
						<span class="side side-{l.team}">{l.team} side</span>
						<span class="result" class:won={r.winner === l.team}>
							{r.winner === l.team ? 'win' : 'loss'}
						</span>
					</span>
					<span class="champs">
						{r.participants
							.filter((p) => p.team === l.team)
							.map((p) => championName(p.champion))
							.join(' · ')}
					</span>
					<span class="q">{l.question}</span>
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
	}
	ul {
		list-style: none;
	}
	a {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 8px;
		color: var(--text);
	}
	a:hover {
		border-color: var(--accent);
		color: var(--text);
	}
	.row {
		display: flex;
		gap: 10px;
		align-items: baseline;
	}
	.t {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 1.2rem;
		font-weight: 600;
	}
	.side,
	.result {
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
	.champs {
		font-size: 0.85rem;
		color: var(--text-muted);
	}
	.q {
		font-size: 0.95rem;
	}
</style>
