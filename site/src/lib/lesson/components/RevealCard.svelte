<script lang="ts">
	import type { AnchorCandidate, PretestChoice } from '$lib/lesson/schema';
	import AnchorMomentCard from './AnchorMomentCard.svelte';

	export let anchor: AnchorCandidate;
	export let userChoice: PretestChoice | null;
	export let cohortProse: string | null;
	export let onContinue: () => void;

	$: timestamp = `${anchor.minute.toString().padStart(2, '0')}:00`;
	$: wins = Math.round(anchor.cohort_winrate * anchor.cohort_size);
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
			<p>
				From where your team stood at {timestamp}, {wins} of {anchor.cohort_size} similar games
				ended in a win{#if anchor.polarity === 'mistake'}. Yours didn't.{:else}, and yours was one of
					them.{/if}
			</p>
			<p class="fallback">
				We can find this moment, but we can't yet say what separated the games that won from
				the ones that didn't.
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
