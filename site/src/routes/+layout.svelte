<script>
	import Menu from '$lib/menu.svelte';
	import '$lib/app.css';
</script>

<div class="layout">
	<nav class="sidebar">
		<Menu />
	</nav>
	<main class="content">
		<slot />
	</main>
</div>

<style>
	.layout {
		display: flex;
		min-height: 100vh;
	}
	.sidebar {
		width: 200px;
		flex-shrink: 0;
		background: var(--bg-surface);
		border-right: 1px solid var(--border);
		padding: 24px 16px;
	}
	.content {
		flex: 1;
		padding: 32px;
		min-width: 0;
	}
	@media (max-width: 640px) {
		.layout {
			flex-direction: column;
		}
		.sidebar {
			width: 100%;
			border-right: none;
			border-bottom: 1px solid var(--border);
			padding: 12px 16px;
		}
		.content {
			padding: 20px 16px;
		}
	}
</style>
