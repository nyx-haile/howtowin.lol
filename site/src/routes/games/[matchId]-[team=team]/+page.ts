import { error } from '@sveltejs/kit';
import { getSampleLesson } from '$lib/lesson/samples';
import type { Team } from '$lib/lesson/schema';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ params }) => {
	const team = params.team as Team;
	const lesson = getSampleLesson(params.matchId, team);
	if (!lesson) throw error(404, `No sample lesson for ${params.matchId} (${team} side)`);
	return { matchId: params.matchId, team, lesson };
};
