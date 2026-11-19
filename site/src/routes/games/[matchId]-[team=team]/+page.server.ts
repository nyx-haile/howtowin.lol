import { error } from '@sveltejs/kit';
import { fetchLesson } from '$lib/server/lesson';
import type { Team } from '$lib/lesson/schema';
import type { PageServerLoad } from './$types';

export const prerender = false;

// Riot match ids look like NA1_5439588777.
const MATCH_ID = /^[A-Z0-9]{2,6}_\d{6,14}$/;

export const load: PageServerLoad = async ({ params }) => {
	if (!MATCH_ID.test(params.matchId)) {
		throw error(404, `Not a match id: ${params.matchId}`);
	}
	const team = params.team as Team;
	const result = await fetchLesson(params.matchId, team);
	return { matchId: params.matchId, team, result };
};
