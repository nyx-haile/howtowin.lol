import type { Side } from './types';

/**
 * A coaching moment, framed the way a VOD reviewer frames it: pause at a
 * timestamp, ask a map decision, show where the player's team went, then
 * what comparison games did from a similar spot.
 *
 * Everything situational (who is dead, timers, towers, walk times) is
 * derived by the UI from the game's own timeline, so this contract only
 * carries what the timeline can't: which moment, which question, and the
 * comparison.
 */
export interface MapTarget {
	id: string;
	label: string;
	x: number;
	y: number;
}

export interface CoachLesson {
	matchId: string;
	team: Side;
	/** Seconds into the game where the replay pauses. */
	t: number;
	/** How long to play forward after the reveal. */
	playForwardS: number;
	question: string;
	targets: MapTarget[];
	/** Which target the team committed to, if any (null = split up). */
	actualTargetId: string | null;
	/** One sentence on what the team actually did. */
	actualSummary: string;
	/** Short factual recap of what happened next, one line per beat. */
	whatHappened: string[];
	/** From comparison games. Null until the model provides it. */
	comparison: {
		games: number;
		byTarget: { targetId: string; games: number; won: number }[];
	} | null;
	/** True while the comparison numbers are illustrative, not model output. */
	illustrative: boolean;
}
