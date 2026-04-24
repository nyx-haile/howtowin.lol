<script lang="ts">
	export let themes: string[] = [];
</script>

<div class="badge" title="Today mixes themes — interleaving strengthens discrimination (lit review §4.1)">
	<span class="label">today:</span>
	<span class="themes">
		{#each themes as t, i}
			<span class="theme">{t}</span>{#if i < themes.length - 1}<span class="plus"> + </span>{/if}
		{/each}
	</span>
</div>

<style>
	.badge {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		padding: 4px 10px;
		background: var(--bg-surface);
		border: 1px dashed var(--border);
		border-radius: 12px;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.label {
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-size: 0.7rem;
	}
	.theme {
		color: var(--text);
		font-weight: 500;
	}
	.plus {
		color: var(--text-muted);
	}
</style>
