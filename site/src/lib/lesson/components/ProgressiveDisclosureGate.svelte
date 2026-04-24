<script lang="ts">
	import { onboardingStore } from '$lib/lesson/stores/onboardingStore';

	export let requiredLessons: number = 1;
	export let label: string = 'Advanced layers';

	$: revealed = $onboardingStore.lessonsCompleted >= requiredLessons;
	$: remaining = Math.max(0, requiredLessons - $onboardingStore.lessonsCompleted);
</script>

{#if revealed}
	<slot />
{:else}
	<div class="gate">
		<span class="label">{label}</span>
		<p>
			Unlocks after {remaining} more committed intent{remaining === 1 ? '' : 's'}. Progressive
			disclosure: Layers 4–6 surface only once Layers 2–3 are a stable habit.
		</p>
	</div>
{/if}

<style>
	.gate {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 12px 14px;
		background: var(--bg-surface);
		border: 1px dashed var(--border);
		border-radius: 6px;
		opacity: 0.7;
	}
	.label {
		text-transform: uppercase;
		font-size: 0.7rem;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}
	p {
		color: var(--text);
		font-size: 0.85rem;
		line-height: 1.5;
	}
</style>
