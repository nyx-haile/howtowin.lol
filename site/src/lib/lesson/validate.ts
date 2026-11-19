import type { AnchorCandidate, LessonResult } from '$lib/lesson/schema';

/** Runtime guard for JSON coming from model.lesson. Mirrors schema.ts. */
export function isLessonResult(x: unknown): x is LessonResult {
	if (!x || typeof x !== 'object') return false;
	const r = x as Record<string, unknown>;
	return (
		typeof r.match_id === 'string' &&
		(r.team === 'blue' || r.team === 'red') &&
		typeof r.player_won === 'boolean' &&
		isAnchorOrNull(r.mistake_anchor) &&
		isAnchorOrNull(r.strength_anchor) &&
		typeof r.checkpoint_sha === 'string' &&
		typeof r.index_checkpoint_sha === 'string' &&
		typeof r.n_mid_anchors === 'number'
	);
}

function isAnchorOrNull(x: unknown): x is AnchorCandidate | null {
	if (x === null) return true;
	if (!x || typeof x !== 'object') return false;
	const a = x as Record<string, unknown>;
	return (
		typeof a.minute === 'number' &&
		typeof a.cohort_entropy === 'number' &&
		typeof a.cohort_winrate === 'number' &&
		typeof a.cohort_size === 'number' &&
		Array.isArray(a.cohort_match_ids) &&
		(a.polarity === 'mistake' || a.polarity === 'strength')
	);
}
