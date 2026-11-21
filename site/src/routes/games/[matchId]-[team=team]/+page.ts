import { error } from '@sveltejs/kit';
import { getLesson, sampleReplays } from '$lib/game/samples';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ params }) => {
	const lesson = getLesson(params.matchId, params.team);
	const replay = sampleReplays[params.matchId];
	if (!lesson || !replay) throw error(404, `No lesson for ${params.matchId} (${params.team} side)`);
	return { lesson, replay };
};
