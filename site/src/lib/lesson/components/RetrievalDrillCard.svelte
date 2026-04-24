<script lang="ts">
	export let scaffoldLevel: 0 | 1 | 2 = 0;
</script>

<section class="drill" data-scaffold={scaffoldLevel}>
	<header>
		<h3>Retrieval drill</h3>
		<span class="tag">stub</span>
	</header>
	<p class="prompt">
		Recall a recent mid-game frame you reviewed. Before the reveal — what would you do?
	</p>
	<p class="note">
		Layer 5: a short, cued retrieval check served the day after a lesson. Wires up in a later
		slice (Roediger & Butler 2011; testing effect).
	</p>
</section>

<style>
	.drill {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px dashed var(--border);
		border-radius: 6px;
		opacity: 0.72;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}
	h3 {
		font-size: 0.95rem;
	}
	.tag {
		text-transform: uppercase;
		font-size: 0.7rem;
		letter-spacing: 0.06em;
		color: var(--text-muted);
	}
	.prompt {
		color: var(--text);
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.note {
		color: var(--text-muted);
		font-size: 0.8rem;
		line-height: 1.5;
	}
</style>
