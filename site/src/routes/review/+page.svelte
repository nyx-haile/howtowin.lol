<script lang="ts">
	import { goto } from '$app/navigation';
	import MomentCard from '$lib/MomentCard.svelte';
	import ConceptBadge from '$lib/ConceptBadge.svelte';

	export let data;

	let query = data.query || '';

	function handleSearch() {
		const trimmed = query.trim();
		if (trimmed) goto(`/review?q=${encodeURIComponent(trimmed)}`);
	}

	function selectGame(matchId: string) {
		goto(`/review?q=${encodeURIComponent(query)}&match=${matchId}`);
	}

	function formatDuration(s: number) {
		return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
	}

	$: review = data.review as {
		player: { riot_id: string; rank_tier: string; rank_division: string };
		games: Array<{
			match_id: string; role: string; win: boolean; duration_s: number;
			key_moments: Array<{ concept_name: string; category: string }>;
		}>;
		top_concepts: Array<{ name: string; explanation: string; category: string; importance: number; times_relevant: number }>;
	} | null;

	$: gameReview = data.gameReview as {
		won: boolean;
		role: string;
		game: { game_duration_s: number; patch: string };
		moments: Array<{
			timestamp_ms: number; concept_name: string; category: string;
			description: string; explanation: string;
		}>;
	} | null;
</script>

<div class="review-page">
	<form class="search" on:submit|preventDefault={handleSearch}>
		<input type="text" bind:value={query} placeholder="gameName#tagLine" />
		<button type="submit">Search</button>
	</form>

	{#if data.error}
		<p class="error">{data.error}</p>
	{/if}

	{#if review}
		<div class="player-header">
			<h2>{review.player.riot_id || 'Unknown'}</h2>
			{#if review.player.rank_tier}
				<span class="rank">{review.player.rank_tier} {review.player.rank_division}</span>
			{/if}
		</div>

		<!-- Top concepts to learn -->
		{#if review.top_concepts && review.top_concepts.length > 0}
			<section class="concepts-to-learn">
				<h3>Your learning focus</h3>
				<p class="section-sub">These concepts came up the most in your recent games.</p>
				<div class="concept-list">
					{#each review.top_concepts as concept}
						<div class="concept-item">
							<ConceptBadge name={concept.name} category={concept.category} />
							<p>{concept.explanation}</p>
							<span class="relevance">Appeared in {concept.times_relevant} of your games</span>
						</div>
					{/each}
				</div>
			</section>
		{/if}

		<!-- Game lesson view -->
		{#if gameReview}
			<section class="game-lesson">
				<div class="lesson-header">
					<h3>Game Review</h3>
					<span class="outcome" class:win={gameReview.won} class:loss={!gameReview.won}>
						{gameReview.won ? 'Victory' : 'Defeat'}
					</span>
					<span class="meta">
						{gameReview.role} &middot; {formatDuration(gameReview.game.game_duration_s)}
					</span>
				</div>

				{#if gameReview.moments.length > 0}
					<p class="lesson-intro">
						Here are the moments that decided this game. Each one is a concept you can learn to spot in future games.
					</p>
					<div class="moments">
						{#each gameReview.moments as moment}
							<MomentCard
								timestamp_ms={moment.timestamp_ms}
								concept_name={moment.concept_name}
								category={moment.category}
								description={moment.description}
								explanation={moment.explanation}
							/>
						{/each}
					</div>
				{:else}
					<p class="muted">No key moments identified for this game yet.</p>
				{/if}
			</section>

		<!-- Game list -->
		{:else}
			<section class="games">
				<h3>Your recent games</h3>
				<p class="section-sub">Pick a game to review. Each game is a lesson.</p>
				<div class="game-list">
					{#each review.games as game}
						<button
							class="game-row"
							class:win={game.win}
							class:loss={!game.win}
							on:click={() => selectGame(game.match_id)}
						>
							<span class="outcome-pip">{game.win ? 'W' : 'L'}</span>
							<span class="role">{game.role}</span>
							<span class="duration">{formatDuration(game.duration_s)}</span>
							<span class="moment-tags">
								{#each game.key_moments.slice(0, 2) as m}
									<ConceptBadge name={m.concept_name} category={m.category} />
								{/each}
							</span>
						</button>
					{/each}
				</div>
			</section>
		{/if}
	{/if}
</div>

<style>
	.review-page { max-width: 680px; }
	.search {
		display: flex; gap: 8px; margin-bottom: 24px;
	}
	input {
		flex: 1; padding: 10px 14px; border-radius: 8px;
		border: 1px solid var(--border); background: var(--bg-surface);
		color: var(--text); font-size: 1rem; outline: none;
	}
	input:focus { border-color: var(--accent); }
	button[type="submit"] {
		padding: 10px 20px; border-radius: 8px; border: none;
		background: var(--accent); color: white; font-weight: 600; cursor: pointer;
	}
	.error { color: var(--loss); margin-bottom: 16px; }
	.player-header {
		display: flex; align-items: baseline; gap: 12px; margin-bottom: 24px;
	}
	h2 { font-size: 1.4rem; }
	.rank { color: var(--text-muted); font-size: 0.9rem; }
	h3 { margin-bottom: 4px; font-size: 1.1rem; }
	.section-sub { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 12px; }
	.concepts-to-learn { margin-bottom: 32px; }
	.concept-list { display: flex; flex-direction: column; gap: 10px; }
	.concept-item {
		background: var(--bg-surface); border: 1px solid var(--border);
		border-radius: 8px; padding: 14px;
	}
	.concept-item p { font-size: 0.9rem; margin-top: 8px; line-height: 1.5; }
	.relevance { font-size: 0.8rem; color: var(--text-muted); display: block; margin-top: 6px; }
	.game-list { display: flex; flex-direction: column; gap: 4px; }
	.game-row {
		display: flex; gap: 12px; padding: 12px 14px; border-radius: 6px;
		background: var(--bg-surface); border: 1px solid var(--border);
		color: var(--text); align-items: center; cursor: pointer;
		text-align: left; width: 100%; font: inherit;
	}
	.game-row:hover { background: var(--bg-hover); }
	.outcome-pip { font-weight: 700; width: 20px; }
	.win .outcome-pip { color: var(--win); }
	.loss .outcome-pip { color: var(--loss); }
	.role { width: 40px; color: var(--text-muted); font-size: 0.85rem; }
	.duration { color: var(--text-muted); font-size: 0.85rem; width: 40px; }
	.moment-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
	.game-lesson { margin-top: 16px; }
	.lesson-header {
		display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px;
	}
	.outcome { font-weight: 700; }
	.win { color: var(--win); }
	.loss { color: var(--loss); }
	.meta { color: var(--text-muted); font-size: 0.85rem; }
	.lesson-intro { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 16px; }
	.moments { display: flex; flex-direction: column; gap: 12px; }
	.muted { color: var(--text-muted); }
</style>
