<script lang="ts">
	import type { AnchorCandidate, PretestChoice } from '$lib/lesson/schema';
	import AnchorMomentCard from './AnchorMomentCard.svelte';

	export let anchor: AnchorCandidate;
	export let userChoice: PretestChoice | null;
	export let cohortProse: string | null;
	export let onContinue: () => void;
</script>

<section class="reveal">
	<AnchorMomentCard {anchor} />

	{#if userChoice}
		<div class="user-choice">
			<span class="tag">you said</span>
			<span class="text">{userChoice.label}</span>
		</div>
	{/if}

	<div class="prose">
		{#if cohortProse}
			<p>{cohortProse}</p>
		{:else}
			<p class="fallback">
				The decision-diff pipeline will fill in what separated cohort winners from losers here.
				For now: cohort entropy {Math.round(anchor.cohort_entropy * 100)}% — this was a pivotal
				minute.
			</p>
		{/if}
	</div>

	<button class="continue" type="button" on:click={onContinue}>
		Set an intent for next game →
	</button>
</section>

<style>
	.reveal {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	.user-choice {
		display: flex;
		gap: 10px;
		padding: 10px 14px;
		background: var(--bg-surface);
		border: 1px dashed var(--border);
		border-radius: 6px;
		font-size: 0.95rem;
	}
	.tag {
		text-transform: uppercase;
		font-size: 0.7rem;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}
	.text {
		color: var(--text);
	}
	.prose p {
		line-height: 1.6;
	}
	.fallback {
		color: var(--text-muted);
		font-size: 0.9rem;
	}
	.continue {
		align-self: flex-start;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 10px 16px;
		cursor: pointer;
		font: inherit;
	}
	.continue:hover {
		background: var(--accent-dim);
	}
</style>
