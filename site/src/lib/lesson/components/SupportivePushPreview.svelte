<script lang="ts">
	// Tone demo for Layer 6. Three notification copies side-by-side to
	// make the "supportive, not guilt-tripping" stance concrete (Renaud 2024,
	// lit review §4.6). Pure presentation — nothing is scheduled or sent.
	const samples = [
		{
			tone: 'supportive',
			copy: 'Played a rough one? Spend 90 seconds on the mid-game frame — we\'ll make it specific.',
		},
		{
			tone: 'curious',
			copy: 'Your last intent was about vision. Want to see whether this game\'s anchor fits the same theme?',
		},
		{
			tone: 'avoid (banned)',
			copy: '🔥 Your streak is at risk! Open now to save it.',
			banned: true,
		},
	];
</script>

<section class="push">
	<header>
		<h3>Notification tone preview</h3>
		<span class="tag">stub</span>
	</header>
	<ul>
		{#each samples as s}
			<li class="sample" class:banned={s.banned}>
				<span class="label">{s.tone}</span>
				<p>{s.copy}</p>
			</li>
		{/each}
	</ul>
</section>

<style>
	.push {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		opacity: 0.8;
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
	ul {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.sample {
		padding: 10px 12px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 6px;
	}
	.sample .label {
		display: block;
		text-transform: uppercase;
		font-size: 0.7rem;
		letter-spacing: 0.06em;
		color: var(--text-muted);
		margin-bottom: 4px;
	}
	.sample p {
		font-size: 0.9rem;
		line-height: 1.5;
		color: var(--text);
	}
	.sample.banned {
		border-color: var(--loss);
		opacity: 0.6;
	}
	.sample.banned p {
		text-decoration: line-through;
		color: var(--text-muted);
	}
	.sample.banned .label {
		color: var(--loss);
	}
</style>
