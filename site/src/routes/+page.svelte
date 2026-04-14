<script>
	import { goto } from '$app/navigation';

	let query = '';

	function handleSearch() {
		const trimmed = query.trim();
		if (trimmed) {
			goto(`/review?q=${encodeURIComponent(trimmed)}`);
		}
	}
</script>

<div class="splash">
	<h1>howtowin<span class="dot">.lol</span></h1>

	<form class="search" on:submit|preventDefault={handleSearch}>
		<input
			type="text"
			bind:value={query}
			placeholder="Search summoner e.g. chaos#kotic"
			autocomplete="off"
		/>
		<button type="submit">
			<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
				<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
			</svg>
		</button>
	</form>
</div>

<style>
	.splash {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 60vh;
		text-align: center;
		gap: 24px;
	}
	h1 {
		font-size: 2.4rem;
		font-weight: 800;
		letter-spacing: -0.04em;
	}
	.dot {
		color: var(--accent);
	}
	.search {
		display: flex;
		width: 100%;
		max-width: 460px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 8px;
		overflow: hidden;
		transition: border-color 0.15s;
	}
	.search:focus-within {
		border-color: var(--accent);
	}
	input {
		flex: 1;
		padding: 14px 16px;
		border: none;
		background: transparent;
		color: var(--text);
		font-size: 0.95rem;
		outline: none;
	}
	input::placeholder {
		color: var(--text-muted);
	}
	button {
		padding: 0 16px;
		border: none;
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
		display: flex;
		align-items: center;
	}
	button:hover {
		color: var(--accent);
	}
</style>
