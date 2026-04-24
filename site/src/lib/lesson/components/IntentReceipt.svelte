<script lang="ts">
	export let intent: string;

	let copied = false;
	let copyTimer: ReturnType<typeof setTimeout> | null = null;

	async function copyIntent() {
		try {
			await navigator.clipboard.writeText(intent);
			copied = true;
			if (copyTimer) clearTimeout(copyTimer);
			copyTimer = setTimeout(() => (copied = false), 1800);
		} catch {
			// clipboard permissions denied; silent fallback
		}
	}
</script>

<section class="receipt">
	<header>
		<span class="tag">committed</span>
		<h3>Next game, you said you'd:</h3>
	</header>
	<blockquote>{intent}</blockquote>
	<footer>
		<button type="button" class="copy" on:click={copyIntent}>
			{copied ? 'Copied ✓' : 'Copy intent'}
		</button>
		<p class="note">Streak advanced. See you after your next game.</p>
	</footer>
</section>

<style>
	.receipt {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 18px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-left: 3px solid var(--accent);
		border-radius: 6px;
	}
	header {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.tag {
		align-self: flex-start;
		text-transform: uppercase;
		font-size: 0.7rem;
		letter-spacing: 0.08em;
		color: var(--accent);
	}
	h3 {
		font-size: 1rem;
		color: var(--text-muted);
		font-weight: 500;
	}
	blockquote {
		margin: 0;
		padding: 12px 14px;
		background: var(--bg);
		border-radius: 6px;
		font-size: 1rem;
		line-height: 1.5;
		color: var(--text);
	}
	footer {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}
	.copy {
		background: transparent;
		color: var(--accent);
		border: 1px solid var(--accent);
		border-radius: 6px;
		padding: 6px 12px;
		cursor: pointer;
		font: inherit;
		font-size: 0.85rem;
	}
	.copy:hover {
		background: color-mix(in srgb, var(--accent) 10%, transparent);
	}
	.note {
		color: var(--text-muted);
		font-size: 0.85rem;
	}
</style>
