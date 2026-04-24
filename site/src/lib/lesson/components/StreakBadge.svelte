<script lang="ts">
	import { streakStore } from '$lib/lesson/stores/streakStore';

	$: count = $streakStore.count;
	$: lastISO = $streakStore.lastMeaningfulISO;
	$: lastLabel = lastISO ? formatRelative(lastISO) : null;

	function formatRelative(iso: string): string {
		const ms = Date.now() - new Date(iso).getTime();
		const days = Math.floor(ms / (1000 * 60 * 60 * 24));
		if (days === 0) return 'today';
		if (days === 1) return 'yesterday';
		return `${days}d ago`;
	}
</script>

<div class="streak" title="Streak advances only when you commit an intent after a lesson.">
	<div class="count-wrap">
		<span class="count">{count}</span>
		<span class="unit">intent{count === 1 ? '' : 's'}</span>
	</div>
	<div class="meta">
		<span class="label">meaningful streak</span>
		{#if lastLabel}
			<span class="last">last: {lastLabel}</span>
		{:else}
			<span class="last">commit one to begin</span>
		{/if}
	</div>
</div>

<style>
	.streak {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		padding: 8px 14px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
	}
	.count-wrap {
		display: flex;
		align-items: baseline;
		gap: 4px;
	}
	.count {
		font-size: 1.4rem;
		font-weight: 700;
		color: var(--accent);
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
	}
	.unit {
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.meta {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.label {
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}
	.last {
		font-size: 0.8rem;
		color: var(--text);
	}
</style>
