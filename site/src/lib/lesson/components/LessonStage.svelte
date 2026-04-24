<script lang="ts">
	import { onMount } from 'svelte';
	import type { LessonViewModel, PretestChoice } from '$lib/lesson/schema';
	import PretestCard from './PretestCard.svelte';
	import RevealCard from './RevealCard.svelte';
	import LessonUnavailable from './LessonUnavailable.svelte';
	import InterventionPrompt from './InterventionPrompt.svelte';
	import IntentReceipt from './IntentReceipt.svelte';
	import { createLessonSession, type LessonSessionStore } from '$lib/lesson/stores/lessonSession';
	import { streakStore } from '$lib/lesson/stores/streakStore';
	import { onboardingStore } from '$lib/lesson/stores/onboardingStore';

	export let viewModel: LessonViewModel;
	export let fixtureId: string;

	let session: LessonSessionStore;
	$: {
		session = createLessonSession(fixtureId);
		if (!viewModel.hasAnchor) {
			session.markNoAnchor();
		}
	}

	$: selectedChoice = findChoice(viewModel.pretestChoices, $session.pretestChoice);

	function findChoice(
		choices: PretestChoice[],
		id: string | null,
	): PretestChoice | null {
		if (!id) return null;
		return choices.find((c) => c.id === id) ?? null;
	}

	function handlePretestCommit(choiceId: string) {
		session.selectPretest(choiceId);
		session.advanceToReveal();
	}

	function handleRevealContinue() {
		session.advanceToIntervention();
	}

	function handleIntentCommit(intent: string) {
		session.commitIntent(intent);
		streakStore.recordIntervention(intent);
		onboardingStore.recordLessonCompleted();
	}

	onMount(() => {
		if (!viewModel.hasAnchor) session.markNoAnchor();
	});
</script>

<div class="stage">
	{#if $session.phase === 'no-anchor' || !viewModel.anchor}
		<LessonUnavailable
			nMidAnchors={viewModel.raw.n_mid_anchors}
			playerWon={viewModel.raw.player_won}
		/>
	{:else if $session.phase === 'pretest'}
		<PretestCard
			anchor={viewModel.anchor}
			choices={viewModel.pretestChoices}
			selectedId={$session.pretestChoice}
			onCommit={handlePretestCommit}
		/>
	{:else if $session.phase === 'reveal'}
		<RevealCard
			anchor={viewModel.anchor}
			userChoice={selectedChoice}
			cohortProse={viewModel.cohortProse}
			onContinue={handleRevealContinue}
		/>
	{:else if $session.phase === 'intervention'}
		<InterventionPrompt
			suggestions={viewModel.interventionSuggestions}
			onAccept={handleIntentCommit}
		/>
	{:else if $session.phase === 'committed' && $session.committedIntent}
		<IntentReceipt intent={$session.committedIntent} />
	{/if}
</div>

<style>
	.stage {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}
</style>
