<script lang="ts">
	import type { AnchorCandidate, PretestChoice } from '$lib/lesson/schema';
	import AnchorMomentCard from './AnchorMomentCard.svelte';

	export let anchor: AnchorCandidate;
	export let choices: PretestChoice[];
	export let selectedId: string | null = null;
	export let onCommit: (choiceId: string) => void;

	function pick(id: string) {
		onCommit(id);
	}
</script>

<section class="pretest">
	<AnchorMomentCard {anchor} />
	<div class="prompt">
		<h3>Before we show what happened — what would you do?</h3>
		<p class="sub">Commit an answer. We'll compare it to what similar games did next.</p>
	</div>
	<ul class="choices">
		{#each choices as c (c.id)}
			<li>
				<button
					class="choice"
					class:selected={selectedId === c.id}
					type="button"
					on:click={() => pick(c.id)}
				>
					{c.label}
				</button>
			</li>
		{/each}
	</ul>
	{#if choices.length === 0}
		<p class="empty">No pretest choices — the decision-diff pipeline hasn't authored options for this anchor yet.</p>
	{/if}
</section>

<style>
	.pretest {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	.prompt h3 {
		font-size: 1.05rem;
		margin-bottom: 4px;
	}
	.prompt .sub {
		color: var(--text-muted);
		font-size: 0.9rem;
	}
	.choices {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.choice {
		width: 100%;
		text-align: left;
		background: var(--bg-surface);
		color: var(--text);
		border: 1px solid var(--border);
		padding: 12px 14px;
		border-radius: 6px;
		font: inherit;
		cursor: pointer;
		transition: border-color 0.12s ease;
	}
	.choice:hover {
		border-color: var(--accent);
	}
	.choice.selected {
		border-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 12%, var(--bg-surface));
	}
	.empty {
		color: var(--text-muted);
		font-size: 0.9rem;
		font-style: italic;
	}
</style>
