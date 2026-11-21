import type { Frame, GameEvent, GameReplay, Participant, Side } from './types';
import { TOWERS, respawnSeconds, type TowerSpot } from './rift';

export interface ChampionState {
	participant: Participant;
	x: number;
	y: number;
	level: number;
	gold: number;
	cs: number;
	/** Seconds until respawn, or 0 if alive. */
	deadFor: number;
	/** Seconds since this position was observed (positions are sampled per minute). */
	positionAgeS: number;
}

export interface GameState {
	t: number;
	champions: ChampionState[];
	towers: (TowerSpot & { alive: boolean })[];
	goldDiff: Record<Side, number>;
	objectives: ObjectiveTimer[];
}

export interface ObjectiveTimer {
	key: 'dragon' | 'herald' | 'baron' | 'grubs';
	label: string;
	status: 'up' | 'spawning' | 'taken' | 'gone';
	/** Seconds until it spawns (status spawning). */
	inS?: number;
	takenBy?: Side;
}

function frameAt(replay: GameReplay, t: number): Frame {
	let best = replay.frames[0];
	for (const f of replay.frames) if (f.t <= t) best = f;
	return best;
}

export function participantById(replay: GameReplay, id: number): Participant | undefined {
	return replay.participants.find((p) => p.id === id);
}

/**
 * Reconstruct what a coach would pause on at time t: where everyone was,
 * who was dead and for how long, towers standing, objective clocks.
 *
 * Positions come from the last per-minute frame, overridden by any kill
 * event (which carries an exact position) involving that champion in the
 * preceding 20 seconds.
 */
export function stateAt(replay: GameReplay, t: number): GameState {
	const frame = frameAt(replay, t);
	const kills = replay.events.filter(
		(e): e is Extract<GameEvent, { type: 'kill' }> => e.type === 'kill' && e.t <= t
	);

	const champions = replay.participants.map((participant) => {
		const fp = frame.p.find((p) => p.id === participant.id)!;
		let x = fp.x;
		let y = fp.y;
		let positionAgeS = t - frame.t;
		for (const k of kills) {
			const involved =
				k.killer === participant.id ||
				k.victim === participant.id ||
				k.assists.includes(participant.id);
			if (involved && t - k.t <= 20 && k.t >= frame.t) {
				x = k.x;
				y = k.y;
				positionAgeS = t - k.t;
			}
		}
		let deadFor = 0;
		for (const k of kills) {
			if (k.victim !== participant.id) continue;
			const back = k.t + respawnSeconds(fp.level, k.t);
			if (t >= k.t && t < back) deadFor = back - t;
		}
		return { participant, x, y, level: fp.level, gold: fp.gold, cs: fp.cs, deadFor, positionAgeS };
	});

	const fallen = new Set(
		replay.events
			.filter((e) => e.type === 'building' && e.t <= t)
			.map((e) => {
				const b = e as Extract<GameEvent, { type: 'building' }>;
				return nearestTower(b.x, b.y)?.id;
			})
	);
	const towers = TOWERS.map((tw) => ({ ...tw, alive: !fallen.has(tw.id) }));

	const gold = { blue: 0, red: 0 };
	for (const c of champions) gold[c.participant.team] += c.gold;

	return {
		t,
		champions,
		towers,
		goldDiff: { blue: gold.blue - gold.red, red: gold.red - gold.blue },
		objectives: objectiveClocks(replay, t)
	};
}

function nearestTower(x: number, y: number): TowerSpot | undefined {
	let best: TowerSpot | undefined;
	let bestD = Infinity;
	for (const tw of TOWERS) {
		const d = Math.hypot(tw.x - x, tw.y - y);
		if (d < bestD) {
			bestD = d;
			best = tw;
		}
	}
	return bestD < 800 ? best : undefined;
}

/** Patch-14 spawn rules: dragon 5:00 then +5:00, grubs 5:00–13:45, Herald 14:00, Baron 20:00. */
function objectiveClocks(replay: GameReplay, t: number): ObjectiveTimer[] {
	const taken = (monster: string) =>
		replay.events.filter(
			(e): e is Extract<GameEvent, { type: 'objective' }> =>
				e.type === 'objective' && e.monster === monster && e.t <= t
		);

	const drakes = taken('DRAGON');
	const lastDrake = drakes.at(-1);
	const nextDrake = lastDrake ? lastDrake.t + 300 : 300;
	const dragon: ObjectiveTimer =
		t >= nextDrake
			? { key: 'dragon', label: 'Dragon', status: 'up' }
			: { key: 'dragon', label: 'Dragon', status: 'spawning', inS: nextDrake - t };

	const heraldKill = taken('RIFTHERALD').at(-1);
	const herald: ObjectiveTimer = heraldKill
		? { key: 'herald', label: 'Rift Herald', status: 'taken', takenBy: heraldKill.team }
		: t < 840
			? { key: 'herald', label: 'Rift Herald', status: 'spawning', inS: 840 - t }
			: t < 1185
				? { key: 'herald', label: 'Rift Herald', status: 'up' }
				: { key: 'herald', label: 'Rift Herald', status: 'gone' };

	const baron: ObjectiveTimer =
		t < 1200
			? { key: 'baron', label: 'Baron', status: 'spawning', inS: 1200 - t }
			: { key: 'baron', label: 'Baron', status: 'up' };

	return [dragon, herald, baron];
}

export function goldDiffSeries(replay: GameReplay, side: Side): { t: number; diff: number }[] {
	return replay.frames.map((f) => {
		let mine = 0;
		let theirs = 0;
		for (const p of f.p) {
			const team = participantById(replay, p.id)?.team;
			if (team === side) mine += p.gold;
			else theirs += p.gold;
		}
		return { t: f.t, diff: mine - theirs };
	});
}

export function eventsBetween(replay: GameReplay, from: number, to: number): GameEvent[] {
	return replay.events.filter((e) => e.t > from && e.t <= to);
}

export function clock(t: number): string {
	const m = Math.floor(t / 60);
	const s = Math.floor(t % 60);
	return `${m}:${s.toString().padStart(2, '0')}`;
}
