import type { LessonResult, Team } from '$lib/lesson/schema';
import { getFixture, fixtureSummaries } from '$lib/lesson/fixtures';

/**
 * Lesson source for /games while the UI is developed separately from the
 * model. Uses only the raw LessonResult from each fixture (no hand-authored
 * pretest, prose or intents), so the page shows exactly what the model's
 * output contract can support on its own.
 *
 * Swap point: when the model is wired back in, replace getSampleLesson with
 * a loader that returns a LessonResult for (matchId, team). Nothing else on
 * the page needs to change.
 */
export interface SampleGame {
	matchId: string;
	team: Team;
	summary: string;
	lesson: LessonResult;
}

export const sampleGames: SampleGame[] = fixtureSummaries.map((s) => {
	const lesson = getFixture(s.id)!.lesson;
	return { matchId: lesson.match_id, team: lesson.team, summary: s.summary, lesson };
});

export function getSampleLesson(matchId: string, team: Team): LessonResult | null {
	return sampleGames.find((g) => g.matchId === matchId && g.team === team)?.lesson ?? null;
}
