<div class="menu">
	<a href="/" class="logo">howtowin<span class="dot">.lol</span></a>
	<nav>
		<a href="/">Home</a>
		<a href="/review">Review</a>
	</nav>
	<div class="footer-nav">
		<a href="/terms">Terms</a>
		<a href="/privacy">Privacy</a>
	</div>
</div>

<style>
	.menu {
		display: flex;
		flex-direction: column;
		gap: 16px;
		height: 100%;
	}
	.logo {
		font-size: 1.1rem;
		font-weight: 700;
		color: var(--text);
		letter-spacing: -0.02em;
	}
	.dot {
		color: var(--accent);
	}
	nav {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-top: 8px;
	}
	nav a {
		color: var(--text-muted);
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 0.9rem;
	}
	nav a:hover {
		background: var(--bg-hover);
		color: var(--text);
	}
	.footer-nav {
		margin-top: auto;
		padding-top: 12px;
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.footer-nav a {
		color: var(--text-muted);
		padding: 4px 12px;
		font-size: 0.8rem;
	}
	.footer-nav a:hover { color: var(--text); }
	@media (max-width: 640px) {
		.menu {
			flex-direction: row;
			align-items: center;
			gap: 12px;
		}
		nav {
			flex-direction: row;
			gap: 4px;
			margin-top: 0;
		}
		.footer-nav {
			flex-direction: row;
			margin-top: 0;
			padding-top: 0;
			border-top: none;
			border-left: 1px solid var(--border);
			padding-left: 12px;
		}
	}
</style>
