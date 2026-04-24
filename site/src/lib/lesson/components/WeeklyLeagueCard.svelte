<script lang="ts">
	import ProgressBar from '$lib/ProgressBar.svelte';
	import { myLeague, peerStandings } from '$lib/lesson/fixtures/league_standings';
</script>

<section class="card">
	<header>
		<h3>{myLeague.name}</h3>
		<span class="tag">stub</span>
	</header>
	<p class="sub">
		Weekly cohort of same-rank improvers. Promotion comes from <em>intents committed</em>, not XP or
		open streaks.
	</p>

	<ProgressBar
		label="intents this week"
		current={myLeague.myIntentsThisWeek}
		target={myLeague.weeklyTargetIntents}
		color="var(--accent)"
	/>

	<ol class="standings">
		{#each peerStandings as p (p.name)}
			<li class:me={p.name === 'you'}>
				<span class="name">{p.name}</span>
				<span class="value">{p.intentsThisWeek}</span>
			</li>
		{/each}
	</ol>
</section>

<style>
	.card {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		opacity: 0.78;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}
	h3 {
		font-size: 0.95rem;
	}
	.tag {
		text-transform: uppercase;
		font-size: 0.7rem;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}
	.sub {
		color: var(--text-muted);
		font-size: 0.85rem;
		line-height: 1.5;
	}
	em {
		font-style: italic;
	}
	.standings {
		list-style: none;
		display: flex;
		flex-direction: column;
	}
	.standings li {
		display: flex;
		justify-content: space-between;
		padding: 5px 0;
		border-top: 1px dashed var(--border);
		font-size: 0.88rem;
	}
	.standings li:first-child {
		border-top: none;
	}
	.standings .me {
		color: var(--accent);
		font-weight: 600;
	}
	.value {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
	}
</style>
