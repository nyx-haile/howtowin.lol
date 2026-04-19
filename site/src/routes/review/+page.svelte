<script lang="ts">
	let email = '';
	let submitted = false;

	function handleSubmit() {
		if (email.trim()) {
			submitted = true;
		}
	}
</script>

<svelte:head>
	<title>Review — howtowin.lol</title>
	<meta name="description" content="Post-game AI coaching for League of Legends. Join the waitlist." />
</svelte:head>

<div class="coming-soon">
	<h2>Game review is coming soon</h2>
	<p class="sub">
		We're pending production API access from Riot Games. Drop your email and we'll notify
		you the moment the first Challenger-level lesson lands.
	</p>

	{#if submitted}
		<p class="ok">Thanks — we'll be in touch.</p>
	{:else}
		<form class="waitlist" on:submit|preventDefault={handleSubmit}>
			<input
				type="email"
				bind:value={email}
				placeholder="you@example.com"
				autocomplete="email"
				required
			/>
			<button type="submit">Join waitlist</button>
		</form>
	{/if}

	<section class="preview">
		<h3>What you'll get</h3>
		<ul>
			<li>A short lesson per game, tied to the moments that decided it.</li>
			<li>Concepts mined from Challenger and pro play, not generic tips.</li>
			<li>Progression tracking — you can see the habits you're actually changing.</li>
		</ul>
	</section>
</div>

<style>
	.coming-soon { max-width: 620px; }
	h2 { font-size: 1.6rem; margin-bottom: 12px; }
	.sub { color: var(--text-muted); margin-bottom: 24px; line-height: 1.6; }
	.ok { color: var(--win); font-weight: 600; }
	.waitlist { display: flex; gap: 8px; margin-bottom: 32px; }
	input {
		flex: 1; padding: 10px 14px; border-radius: 8px;
		border: 1px solid var(--border); background: var(--bg-surface);
		color: var(--text); font-size: 1rem; outline: none;
	}
	input:focus { border-color: var(--accent); }
	button {
		padding: 10px 20px; border-radius: 8px; border: none;
		background: var(--accent); color: white; font-weight: 600; cursor: pointer;
	}
	.preview { border-top: 1px solid var(--border); padding-top: 24px; }
	h3 { font-size: 1rem; margin-bottom: 12px; }
	ul { list-style: none; display: flex; flex-direction: column; gap: 8px; }
	li { color: var(--text-muted); font-size: 0.95rem; padding-left: 16px; position: relative; }
	li::before { content: '→'; position: absolute; left: 0; color: var(--accent); }
</style>
