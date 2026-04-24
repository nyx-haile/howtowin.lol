<script lang="ts">
	import type { AnchorCandidate } from '$lib/lesson/schema';
	import CohortSparkline from './CohortSparkline.svelte';
	import { cohortSizeLabel } from '$lib/lesson/epistemic';

	export let anchor: AnchorCandidate;
	export let scaffoldLevel: 0 | 1 | 2 = 0;

	$: mm = anchor.minute.toString().padStart(2, '0');
	$: timestamp = `${mm}:00`;
	$: entropyPct = Math.round(anchor.cohort_entropy * 100);
</script>

<article class="card polarity-{anchor.polarity}" data-scaffold={scaffoldLevel}>
	<header>
		<span class="timestamp">{timestamp}</span>
		<span class="polarity">{anchor.polarity}</span>
	</header>
	<div class="meta">
		<span>{cohortSizeLabel(anchor.cohort_size)}</span>
		<span class="sep">·</span>
		<span>decision weight {entropyPct}%</span>
	</div>
	<CohortSparkline winrate={anchor.cohort_winrate} size={anchor.cohort_size} />
</article>

<style>
	.card {
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-left-width: 3px;
		border-radius: 6px;
		padding: 16px 18px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.polarity-mistake {
		border-left-color: var(--loss);
	}
	.polarity-strength {
		border-left-color: var(--win);
	}
	header {
		display: flex;
		align-items: baseline;
		gap: 12px;
	}
	.timestamp {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 1.15rem;
		color: var(--text);
	}
	.polarity {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}
	.meta {
		display: flex;
		gap: 6px;
		font-size: 0.85rem;
		color: var(--text-muted);
	}
	.sep {
		color: var(--border);
	}
</style>
