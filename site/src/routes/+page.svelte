<script lang="ts">
	import WaitlistForm from '$lib/WaitlistForm.svelte';
	import { goto } from '$app/navigation';

	function openReview() {
		goto('/review');
	}
</script>

<svelte:head>
	<title>howtowin.lol — Duolingo for League of Legends</title>
	<meta
		name="description"
		content="Post-game AI coaching for League of Legends. Every game becomes a bite-sized lesson grounded in Challenger-level decision making."
	/>
</svelte:head>

<div class="landing">
	<header class="hero">
		<h1>howtowin<span class="dot">.lol</span></h1>
		<p class="tag">Duolingo for League of Legends.</p>
		<p class="lede">
			Paste a game. Get a lesson. We mine Challenger and pro play to surface the moments that
			decided your match, and turn each one into a concept you can actually learn.
		</p>

		<WaitlistForm source="landing" marginBottom="0" />
		<button class="secondary" type="button" on:click={openReview}>See a preview</button>
	</header>

	<section class="pillars">
		<div class="pillar">
			<h3>Not another stats site</h3>
			<p>
				We don't show you KDA. We show you the <em>decisions</em> — the wave you should have crashed,
				the fight you should have disengaged — with a named concept you can spot next time.
			</p>
		</div>
		<div class="pillar">
			<h3>Challenger as ground truth</h3>
			<p>
				Our models are trained on high-elo and pro matches. Concepts aren't hand-written tips —
				they're learned from what top players actually do.
			</p>
		</div>
		<div class="pillar">
			<h3>Progression, not doom-scroll</h3>
			<p>
				Each concept is a unit. You practice it, you see it show up in later games, you level up. No
				leaderboards, no toxicity — just the habit loop.
			</p>
		</div>
	</section>

	<section class="status">
		<h3>Where we are</h3>
		<p>
			We're currently in closed development, pending production API access from Riot Games. Drop
			your email above and you'll get a DM as soon as the first public lesson is ready.
		</p>
	</section>

	<footer class="legal">
		<p>
			howtowin.lol isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot
			Games or anyone officially involved in producing or managing League of Legends. League of
			Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc. League of
			Legends © Riot Games, Inc.
		</p>
	</footer>
</div>

<style>
	.landing {
		max-width: 780px;
		margin: 0 auto;
		display: flex;
		flex-direction: column;
		gap: 56px;
	}
	.hero {
		display: flex;
		flex-direction: column;
		gap: 16px;
		align-items: flex-start;
		padding-top: 24px;
	}
	h1 {
		font-size: 2.8rem;
		font-weight: 800;
		letter-spacing: -0.04em;
	}
	.dot {
		color: var(--accent);
	}
	.tag {
		font-size: 1.15rem;
		color: var(--text);
		font-weight: 600;
	}
	.lede {
		color: var(--text-muted);
		line-height: 1.65;
		max-width: 580px;
		font-size: 1rem;
	}
	.secondary {
		background: transparent;
		color: var(--text-muted);
		border: 1px solid var(--border);
		padding: 10px 16px;
		border-radius: 8px;
		cursor: pointer;
		font-size: 0.95rem;
	}
	.secondary:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.pillars {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
		gap: 20px;
	}
	.pillar {
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 20px;
	}
	.pillar h3 {
		font-size: 1rem;
		margin-bottom: 10px;
	}
	.pillar p {
		color: var(--text-muted);
		font-size: 0.9rem;
		line-height: 1.6;
	}
	.status h3 {
		font-size: 1rem;
		margin-bottom: 10px;
	}
	.status p {
		color: var(--text-muted);
		line-height: 1.65;
	}
	.legal {
		border-top: 1px solid var(--border);
		padding-top: 20px;
	}
	.legal p {
		color: var(--text-muted);
		font-size: 0.8rem;
		line-height: 1.6;
	}
</style>
