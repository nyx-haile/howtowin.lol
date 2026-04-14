<script lang="ts">
	import ConceptBadge from './ConceptBadge.svelte';

	export let timestamp_ms: number;
	export let concept_name: string;
	export let category: string;
	export let description: string;
	export let explanation: string;

	$: minute = Math.floor(timestamp_ms / 60000);
	$: seconds = Math.floor((timestamp_ms % 60000) / 1000);
	$: timeStr = `${minute}:${seconds.toString().padStart(2, '0')}`;
</script>

<div class="card">
	<div class="header">
		<span class="time">{timeStr}</span>
		<ConceptBadge name={concept_name} {category} />
	</div>
	<p class="what-happened">{description}</p>
	<details>
		<summary>Why does this matter?</summary>
		<p class="explanation">{explanation}</p>
	</details>
</div>

<style>
	.card {
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 16px;
	}
	.header {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 8px;
	}
	.time {
		font-weight: 700;
		font-size: 0.9rem;
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}
	.what-happened {
		font-size: 0.95rem;
		line-height: 1.5;
		margin-bottom: 8px;
	}
	details {
		font-size: 0.85rem;
		color: var(--text-muted);
	}
	summary {
		cursor: pointer;
		color: var(--accent);
		font-weight: 500;
	}
	summary:hover {
		color: var(--accent-dim);
	}
	.explanation {
		margin-top: 8px;
		line-height: 1.6;
		padding: 12px;
		background: var(--bg-hover);
		border-radius: 6px;
	}
</style>
