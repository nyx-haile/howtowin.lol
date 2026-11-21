import type { GameEvent, GameReplay, Side } from './types';
import { stateAt, type GameState } from './replay';

/**
 * Approximate fog of war from one team's point of view.
 *
 * Riot's timeline records wards placed/killed but not where, so true vision
 * can't be rebuilt. This approximation reveals an enemy only when:
 *  - they are within sight range of a living ally or a standing allied tower, or
 *  - they took part in a kill or objective in the last few seconds.
 * Ward vision is ignored, so it shows the player LESS than they really saw:
 * puzzles err harder, never easier.
 */
export const SIGHT_RANGE = 1350;
const EVENT_REVEAL_S = 4;
const LOOKBACK_S = 120;

export type Visibility =
	| { kind: 'visible' }
	| { kind: 'last-seen'; x: number; y: number; agoS: number }
	| { kind: 'unseen' };

function revealedIn(state: GameState, team: Side, id: number): boolean {
	const enemy = state.champions.find((c) => c.participant.id === id)!;
	if (enemy.deadFor > 0) return true; // death is announced to everyone
	for (const c of state.champions) {
		if (c.participant.team !== team || c.deadFor > 0) continue;
		if (Math.hypot(c.x - enemy.x, c.y - enemy.y) <= SIGHT_RANGE) return true;
	}
	for (const tw of state.towers) {
		if (tw.team !== team || !tw.alive) continue;
		if (Math.hypot(tw.x - enemy.x, tw.y - enemy.y) <= SIGHT_RANGE) return true;
	}
	return false;
}

function inEvent(e: GameEvent, id: number): boolean {
	if (e.type === 'kill') return e.killer === id || e.victim === id || e.assists.includes(id);
	return e.killer === id;
}

export function visibilityAt(
	replay: GameReplay,
	t: number,
	team: Side,
	state: GameState = stateAt(replay, t)
): Map<number, Visibility> {
	const out = new Map<number, Visibility>();
	const enemies = replay.participants.filter((p) => p.team !== team);

	for (const p of enemies) {
		const recent = replay.events.some(
			(e) => e.t <= t && t - e.t <= EVENT_REVEAL_S && inEvent(e, p.id)
		);
		if (recent || revealedIn(state, team, p.id)) {
			out.set(p.id, { kind: 'visible' });
			continue;
		}
		// Walk back through moments we have data for, newest first.
		const moments = [
			...replay.frames.map((f) => f.t),
			...replay.events.filter((e) => inEvent(e, p.id)).map((e) => e.t)
		]
			.filter((m) => m < t && t - m <= LOOKBACK_S)
			.sort((a, b) => b - a);
		let found: Visibility = { kind: 'unseen' };
		for (const m of moments) {
			const s = stateAt(replay, m);
			const ev = replay.events.some((e) => Math.abs(e.t - m) < 0.5 && inEvent(e, p.id));
			if (ev || revealedIn(s, team, p.id)) {
				const c = s.champions.find((c) => c.participant.id === p.id)!;
				found = { kind: 'last-seen', x: c.x, y: c.y, agoS: t - m };
				break;
			}
		}
		out.set(p.id, found);
	}
	for (const p of replay.participants.filter((p) => p.team === team))
		out.set(p.id, { kind: 'visible' });
	return out;
}
