<script lang="ts">
	export let label: string;
	export let current: number;
	export let target: number;
	export let color: string = 'var(--accent)';

	$: pct = Math.min(100, Math.round((current / Math.max(target, 1)) * 100));
</script>

<div class="progress">
	<div class="label-row">
		<span class="label">{label}</span>
		<span class="count">{current}/{target}</span>
	</div>
	<div class="bar">
		<div class="fill" style="width: {pct}%; background: {color}" />
	</div>
</div>

<style>
	.progress {
		margin-bottom: 12px;
	}
	.label-row {
		display: flex;
		justify-content: space-between;
		margin-bottom: 4px;
	}
	.label {
		font-size: 0.85rem;
		font-weight: 500;
	}
	.count {
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.bar {
		height: 6px;
		background: var(--bg-hover);
		border-radius: 3px;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		border-radius: 3px;
		transition: width 0.3s ease;
	}
</style>
