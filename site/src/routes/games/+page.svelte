<script lang="ts">
	import { goto } from '$app/navigation';

	let matchId = '';
	let team: 'blue' | 'red' = 'blue';

	$: cleaned = matchId.trim().toUpperCase();
	$: valid = /^[A-Z0-9]{2,6}_\d{6,14}$/.test(cleaned);

	function submit() {
		if (valid) goto(`/games/${cleaned}-${team}`);
	}
</script>

<svelte:head>
	<title>Review a game — howtowin.lol</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<form class="lookup" on:submit|preventDefault={submit}>
	<h2>Review a game</h2>
	<p class="sub">
		Pick a match and the side you played. We'll find the minute where similar games split between
		wins and losses.
	</p>

	<label>
		<span>Match id</span>
		<input
			name="matchId"
			placeholder="NA1_5439588777"
			autocomplete="off"
			spellcheck="false"
			bind:value={matchId}
		/>
	</label>

	<fieldset>
		<legend>Your side</legend>
		<label class="radio"><input type="radio" bind:group={team} value="blue" /> Blue</label>
		<label class="radio"><input type="radio" bind:group={team} value="red" /> Red</label>
	</fieldset>

	<button type="submit" disabled={!valid}>Find the moment →</button>
</form>

<style>
	.lookup {
		max-width: 480px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	h2 {
		font-size: 1.4rem;
	}
	.sub {
		color: var(--text-muted);
		line-height: 1.6;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 0.9rem;
		color: var(--text-muted);
	}
	input[name='matchId'] {
		background: var(--bg-surface);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 10px 12px;
		font: inherit;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
	}
	input[name='matchId']:focus {
		outline: none;
		border-color: var(--accent);
	}
	fieldset {
		border: none;
		display: flex;
		gap: 16px;
		align-items: center;
		font-size: 0.9rem;
		color: var(--text-muted);
	}
	legend {
		margin-bottom: 6px;
	}
	.radio {
		flex-direction: row;
		align-items: center;
		gap: 6px;
		color: var(--text);
		cursor: pointer;
	}
	button {
		align-self: flex-start;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 10px 16px;
		cursor: pointer;
		font: inherit;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	button:not(:disabled):hover {
		background: var(--accent-dim);
	}
</style>
