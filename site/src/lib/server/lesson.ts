import { env } from '$env/dynamic/private';
import type { LessonResult, Team } from '$lib/lesson/schema';
import { isLessonResult } from '$lib/lesson/validate';

export type LessonFetch =
	| { ok: true; lesson: LessonResult }
	| { ok: false; reason: 'engine_off' | 'model_unavailable' | 'failed'; detail: string };

/** Cold start loads the checkpoint plus the multi-GB retrieval index. */
const LESSON_TIMEOUT_MS = 90_000;

/**
 * Run (or read from cache) `model.lesson.generate_lesson` for one game.
 *
 * Gated on HOWL_LESSON_ENGINE=1 so the public Cloudflare deploy never tries
 * to spawn Python; set it on the host that has the checkpoint + index.
 */
export async function fetchLesson(matchId: string, team: Team): Promise<LessonFetch> {
	if (env.HOWL_LESSON_ENGINE !== '1') {
		return {
			ok: false,
			reason: 'engine_off',
			detail: 'HOWL_LESSON_ENGINE is not set on this server.'
		};
	}
	try {
		const { queryPython } = await import('$lib/server/python');
		const out = await queryPython<unknown>(
			{ action: 'lesson', match_id: matchId, team },
			{ timeoutMs: LESSON_TIMEOUT_MS }
		);
		if (out && typeof out === 'object' && 'error' in out) {
			const e = out as { error: string; detail?: string };
			return {
				ok: false,
				reason: e.error === 'model_unavailable' ? 'model_unavailable' : 'failed',
				detail: e.detail ?? e.error
			};
		}
		if (!isLessonResult(out)) {
			return { ok: false, reason: 'failed', detail: 'Lesson output did not match LessonResult.' };
		}
		return { ok: true, lesson: out };
	} catch (err) {
		return { ok: false, reason: 'failed', detail: (err as Error).message };
	}
}
