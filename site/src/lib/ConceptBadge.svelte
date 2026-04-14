<script lang="ts">
	export let name: string;
	export let category: string = 'general';

	const COLORS: Record<string, string> = {
		economy: 'var(--concept-economy)',
		objectives: 'var(--concept-objectives)',
		vision: 'var(--concept-vision)',
		fighting: 'var(--concept-fighting)',
		macro: 'var(--concept-macro)',
	};

	$: color = COLORS[category] || 'var(--text-muted)';
</script>

<span class="badge" style="--badge-color: {color}">
	{name}
</span>

<style>
	.badge {
		display: inline-block;
		padding: 2px 10px;
		border-radius: 12px;
		font-size: 0.8rem;
		font-weight: 600;
		background: color-mix(in srgb, var(--badge-color) 15%, transparent);
		color: var(--badge-color);
		border: 1px solid color-mix(in srgb, var(--badge-color) 30%, transparent);
	}
</style>
