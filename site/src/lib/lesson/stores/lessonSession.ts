/** Pretest → reveal → intervention phase machine.
 *
 * Route-scoped: each [fixtureId] creates a fresh store so committing an
 * intent in one fixture doesn't skip the pretest on another. State is not
 * URL-encoded (would leak answers via browser history) and not in page
 * data (can't respond to user interaction). Refresh resets to 'pretest';
 * documented acceptable for v1 stakeholder walkthroughs.
 */
import { writable, type Writable, derived, type Readable } from 'svelte/store';

export type LessonPhase =
	| 'pretest'
	| 'reveal'
	| 'intervention'
	| 'committed'
	| 'no-anchor';

export interface LessonSession {
	fixtureId: string;
	phase: LessonPhase;
	pretestChoice: string | null;
	committedIntent: string | null;
}

export interface LessonSessionStore extends Readable<LessonSession> {
	selectPretest: (choiceId: string) => void;
	advanceToReveal: () => void;
	advanceToIntervention: () => void;
	commitIntent: (intent: string) => void;
	markNoAnchor: () => void;
	reset: () => void;
}

export function createLessonSession(
	fixtureId: string,
	opts: { startAt?: LessonPhase } = {},
): LessonSessionStore {
	const initial: LessonSession = {
		fixtureId,
		phase: opts.startAt ?? 'pretest',
		pretestChoice: null,
		committedIntent: null,
	};
	const store: Writable<LessonSession> = writable(initial);
	return {
		subscribe: store.subscribe,
		selectPretest: (choiceId) =>
			store.update((s) => ({ ...s, pretestChoice: choiceId })),
		advanceToReveal: () =>
			store.update((s) =>
				s.pretestChoice ? { ...s, phase: 'reveal' } : s,
			),
		advanceToIntervention: () =>
			store.update((s) => ({ ...s, phase: 'intervention' })),
		commitIntent: (intent) =>
			store.update((s) => ({
				...s,
				committedIntent: intent,
				phase: 'committed',
			})),
		markNoAnchor: () => store.update((s) => ({ ...s, phase: 'no-anchor' })),
		reset: () => store.set(initial),
	};
}

/** Selector helper: true iff the user has chosen a pretest option. */
export function hasPretestChoice(
	session: LessonSessionStore,
): Readable<boolean> {
	return derived(session, ($s) => $s.pretestChoice !== null);
}
