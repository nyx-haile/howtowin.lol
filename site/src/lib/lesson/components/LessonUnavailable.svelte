<script lang="ts">
	export let nMidAnchors: number;
	export let playerWon: boolean;
</script>

<section class="empty">
	<h3>Nothing to teach from this one.</h3>
	<p>
		{#if playerWon}
			You won, and every mid-game frame looked like a lopsided cohort — no high-entropy decision
			crossed the threshold. That usually means the game was already decided by the map state.
		{:else}
			The loss didn't concentrate in a single decision. Mid-game frames looked lopsided across
			cohort games, so there's no clean counterfactual to show.
		{/if}
	</p>
	<p class="detail">
		{nMidAnchors} mid-game frame{nMidAnchors === 1 ? '' : 's'} inspected.
	</p>
</section>

<style>
	.empty {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 20px 18px;
		background: var(--bg-surface);
		border: 1px dashed var(--border);
		border-radius: 6px;
	}
	h3 {
		font-size: 1.05rem;
	}
	p {
		color: var(--text-muted);
		line-height: 1.6;
	}
	.detail {
		font-size: 0.85rem;
	}
</style>
