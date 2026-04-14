import { queryPython } from '$lib/server/python';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url }) => {
	const q = url.searchParams.get('q');
	const matchId = url.searchParams.get('match');

	if (!q) return { query: '', review: null, gameReview: null, error: null };

	// Search for player
	const players = await queryPython<Array<{ puuid: string; riot_id: string }>>({
		action: 'search',
		query: q
	});

	if (!players || players.length === 0) {
		return { query: q, review: null, gameReview: null, error: 'Player not found. Make sure they have been analyzed.' };
	}

	const puuid = players[0].puuid;

	// Get player review (recent games + concepts)
	const review = await queryPython<Record<string, unknown>>({
		action: 'review',
		puuid
	});

	// If a specific match is selected, get the game lesson
	let gameReview = null;
	if (matchId) {
		gameReview = await queryPython<Record<string, unknown>>({
			action: 'game',
			match_id: matchId,
			puuid
		});
	}

	return { query: q, review, gameReview, error: null };
};
