<script lang="ts">
	import { cohortOutcomeLabel } from '$lib/lesson/epistemic';
	export let winrate: number;
	export let size: number;
	$: winPct = Math.round(winrate * 100);
	$: label = cohortOutcomeLabel(winrate, size);
</script>

<figure>
	<div class="bar" role="img" aria-label={label}>
		<span class="win" style="width: {winPct}%"></span>
		<span class="loss" style="width: {100 - winPct}%"></span>
	</div>
	<figcaption>{label}</figcaption>
</figure>

<style>
	figure {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.bar {
		display: flex;
		height: 8px;
		width: 100%;
		border-radius: 4px;
		overflow: hidden;
		background: var(--border);
	}
	.win {
		background: var(--win);
	}
	.loss {
		background: var(--loss);
	}
	figcaption {
		font-size: 0.85rem;
		color: var(--text-muted);
	}
</style>
