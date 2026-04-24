<script lang="ts">
	export let suggestions: string[] = [];
	export let onAccept: (intent: string) => void;

	let selected: string | null = null;
	let custom = '';

	$: effective = (selected ?? custom).trim();
	$: canCommit = effective.length > 0;

	function pick(s: string) {
		selected = s;
		custom = '';
	}

	function commit() {
		if (!canCommit) return;
		onAccept(effective);
	}
</script>

<section class="intervention">
	<header>
		<h3>One thing you'll do differently next game.</h3>
		<p class="sub">
			Commit a specific, small action. Streak advances only when you commit an intent — not when
			you open the app or read a lesson.
		</p>
	</header>

	{#if suggestions.length > 0}
		<ul class="suggestions">
			{#each suggestions as s}
				<li>
					<button
						type="button"
						class="suggestion"
						class:selected={selected === s}
						on:click={() => pick(s)}
					>
						{s}
					</button>
				</li>
			{/each}
		</ul>
	{/if}

	<label class="custom">
		<span>Or write your own:</span>
		<textarea
			rows="2"
			placeholder="e.g. Ward river bush at 13:30 before pushing wave"
			bind:value={custom}
			on:input={() => (selected = null)}
		></textarea>
	</label>

	<button class="commit" type="button" disabled={!canCommit} on:click={commit}>
		Commit intent
	</button>
</section>

<style>
	.intervention {
		display: flex;
		flex-direction: column;
		gap: 14px;
		padding: 18px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
	}
	header h3 {
		font-size: 1.05rem;
		margin-bottom: 4px;
	}
	header .sub {
		color: var(--text-muted);
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.suggestions {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.suggestion {
		width: 100%;
		text-align: left;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		padding: 10px 12px;
		border-radius: 6px;
		font: inherit;
		cursor: pointer;
		transition: border-color 0.12s ease;
	}
	.suggestion:hover {
		border-color: var(--accent);
	}
	.suggestion.selected {
		border-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 12%, var(--bg));
	}
	.custom {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 0.9rem;
		color: var(--text-muted);
	}
	.custom textarea {
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 10px 12px;
		font: inherit;
		resize: vertical;
	}
	.custom textarea:focus {
		outline: none;
		border-color: var(--accent);
	}
	.commit {
		align-self: flex-start;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 10px 16px;
		cursor: pointer;
		font: inherit;
	}
	.commit:disabled {
		background: var(--border);
		cursor: not-allowed;
	}
	.commit:not(:disabled):hover {
		background: var(--accent-dim);
	}
</style>
