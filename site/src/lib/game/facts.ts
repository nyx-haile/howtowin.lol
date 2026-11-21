import type { GameState, ChampionState } from './replay';
import type { MapTarget } from './lesson';
import type { Side } from './types';
import { walkSeconds, PITS } from './rift';
import { championName } from './champions';
import type { Visibility } from './vision';

/**
 * Coach-style facts about a paused moment, derived only from timeline data.
 * Each fact is something a reviewer would point at on the minimap.
 */
export interface Fact {
	kind: 'down' | 'base' | 'objective' | 'race' | 'fog';
	text: string;
	side?: Side;
}

const FOUNTAIN: Record<Side, { x: number; y: number }> = {
	blue: { x: 394, y: 461 },
	red: { x: 14340, y: 14390 }
};

export function inBase(c: ChampionState): boolean {
	const f = FOUNTAIN[c.participant.team];
	return Math.hypot(c.x - f.x, c.y - f.y) < 2600;
}

export function factsAt(
	state: GameState,
	team: Side,
	targets: MapTarget[],
	visibility?: Map<number, Visibility>
): Fact[] {
	const facts: Fact[] = [];
	const seen = (c: ChampionState) =>
		!visibility || visibility.get(c.participant.id)?.kind === 'visible';
	const enemy: Side = team === 'blue' ? 'red' : 'blue';

	for (const side of [enemy, team]) {
		for (const c of state.champions.filter((c) => c.participant.team === side)) {
			const name = championName(c.participant.champion);
			const v = visibility?.get(c.participant.id);
			if (side === enemy && v && v.kind !== 'visible' && c.deadFor === 0) {
				facts.push({
					kind: 'fog',
					side,
					text:
						v.kind === 'last-seen'
							? `${name}: last seen ${Math.round(v.agoS)}s ago`
							: `${name}: not seen in the last 2 minutes`
				});
				continue;
			}
			if (c.deadFor > 0) {
				facts.push({
					kind: 'down',
					side,
					text: `${name} is dead, back in ≈${Math.ceil(c.deadFor)}s`
				});
			} else if (inBase(c)) {
				facts.push({ kind: 'base', side, text: `${name} is in base` });
			}
		}
	}

	for (const o of state.objectives) {
		if (o.status === 'up') facts.push({ kind: 'objective', text: `${o.label} is up` });
		else if (o.status === 'spawning' && o.inS !== undefined && o.inS < 300)
			facts.push({ kind: 'objective', text: `${o.label} spawns in ${fmt(o.inS)}` });
	}

	// Races only make sense for neutral objectives both teams can contest.
	const pits = Object.values(PITS);
	for (const tg of targets) {
		if (!pits.some((p) => Math.hypot(p.x - tg.x, p.y - tg.y) < 1500)) continue;
		const ours = closest(state, team, tg, seen);
		const theirs = closest(state, enemy, tg, seen);
		if (!ours || !theirs) continue;
		if (Math.min(ours.s, theirs.s) > 30) continue;
		facts.push({
			kind: 'race',
			text: `Closest to ${tg.label}: your ${championName(ours.c.participant.champion)} ≈${Math.round(ours.s)}s, closest ${visibility ? 'visible ' : ''}enemy ${championName(theirs.c.participant.champion)} ≈${Math.round(theirs.s)}s`
		});
	}
	return facts;
}

function closest(state: GameState, side: Side, tg: MapTarget, seen: (c: ChampionState) => boolean) {
	let best: { c: ChampionState; s: number } | null = null;
	for (const c of state.champions) {
		if (c.participant.team !== side || c.deadFor > 0 || inBase(c) || !seen(c)) continue;
		const s = walkSeconds(c, tg);
		if (!best || s < best.s) best = { c, s };
	}
	return best;
}

function fmt(s: number): string {
	const m = Math.floor(s / 60);
	const r = Math.round(s % 60);
	return m > 0 ? `${m}:${r.toString().padStart(2, '0')}` : `${r}s`;
}
