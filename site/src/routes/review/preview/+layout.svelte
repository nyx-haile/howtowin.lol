<script lang="ts">
	// Intentionally minimal. The preview banner is always visible; everything
	// below comes from the child route.
</script>

<svelte:head>
	<title>Preview — howtowin.lol</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<div class="preview-shell">
	<div class="banner" role="note">
		<span class="dot" aria-hidden="true"></span>
		<strong>Preview</strong>
		<span class="sep">·</span>
		<span>mock data, not a live review</span>
	</div>
	<div class="body">
		<slot />
	</div>
</div>

<style>
	.preview-shell {
		display: flex;
		flex-direction: column;
		gap: 20px;
		max-width: 820px;
	}
	.banner {
		display: flex;
		align-items: center;
		gap: 10px;
		background: color-mix(in srgb, var(--gold) 12%, var(--bg-surface));
		border: 1px solid color-mix(in srgb, var(--gold) 40%, var(--border));
		border-radius: 6px;
		padding: 10px 14px;
		font-size: 0.9rem;
		color: var(--text);
	}
	.banner strong {
		color: var(--gold);
		letter-spacing: 0.04em;
		text-transform: uppercase;
		font-size: 0.8rem;
	}
	.banner .sep {
		color: var(--text-muted);
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--gold);
		box-shadow: 0 0 8px var(--gold);
	}
	.body {
		display: flex;
		flex-direction: column;
		gap: 24px;
	}
</style>
