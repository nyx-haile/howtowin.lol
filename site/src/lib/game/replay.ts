import type { Frame, GameEvent, GameReplay, Participant, Side } from './types';
import { TOWERS, respawnSeconds, type TowerSpot } from './rift';

export interface ChampionState {
	participant: Participant;
	x: number;
	y: number;
	level: number;
	gold: number;
	cs: number;
	hp: number;
	hpMax: number;
	mp: number;
	mpMax: number;
	/** Seconds until respawn, or 0 if alive. */
	deadFor: number;
	/** True when position comes from data at (or within 2s of) this moment, false when interpolated. */
	exact: boolean;
	/** True when HP/mana are recorded at this moment (minute mark, death, respawn). */
	vitalsExact: boolean;
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

const FOUNTAIN = { blue: { x: 394, y: 461 }, red: { x: 14340, y: 14390 } };
const EXACT_WINDOW_S = 2;

interface Anchor {
	t: number;
	x: number;
	y: number;
}

/**
 * Known positions for one champion over the whole game: every minute frame,
 * every event that carries their position (kills they were part of), and
 * the fountain when they respawn. Everything between anchors is
 * interpolated and must be shown as approximate.
 */
function anchorsFor(replay: GameReplay, id: number): Anchor[] {
	const team = participantById(replay, id)!.team;
	const out: Anchor[] = replay.frames.map((f) => {
		const p = f.p.find((q) => q.id === id)!;
		return { t: f.t, x: p.x, y: p.y };
	});
	for (const e of replay.events) {
		if (e.type !== 'kill') continue;
		if (e.killer === id || e.victim === id || e.assists.includes(id))
			out.push({ t: e.t, x: e.x, y: e.y });
		if (e.victim === id) {
			const lvl = frameAt(replay, e.t).p.find((q) => q.id === id)!.level;
			const back = e.t + respawnSeconds(lvl, e.t);
			// Body stays where it fell until respawn, then appears at fountain.
			out.push({ t: back - 0.01, x: e.x, y: e.y });
			out.push({ t: back, ...FOUNTAIN[team] });
		}
	}
	return out.sort((a, b) => a.t - b.t);
}

const anchorCache = new WeakMap<GameReplay, Map<number, Anchor[]>>();
function cachedAnchors(replay: GameReplay, id: number): Anchor[] {
	let m = anchorCache.get(replay);
	if (!m) anchorCache.set(replay, (m = new Map()));
	let a = m.get(id);
	if (!a) m.set(id, (a = anchorsFor(replay, id)));
	return a;
}

function positionAt(replay: GameReplay, id: number, t: number) {
	const anchors = cachedAnchors(replay, id);
	let prev = anchors[0];
	let next = anchors[anchors.length - 1];
	for (const a of anchors) {
		if (a.t <= t) prev = a;
		if (a.t >= t) {
			next = a;
			break;
		}
	}
	const exact = Math.min(Math.abs(t - prev.t), Math.abs(next.t - t)) <= EXACT_WINDOW_S;
	if (next.t === prev.t) return { x: prev.x, y: prev.y, exact };
	const k = (t - prev.t) / (next.t - prev.t);
	return { x: prev.x + (next.x - prev.x) * k, y: prev.y + (next.y - prev.y) * k, exact };
}

/**
 * Reconstruct the board at second t: positions (exact at anchors,
 * interpolated between), HP/mana (exact at minute marks; between marks the
 * last recorded value, flagged not exact), deaths and respawns, towers,
 * objective clocks.
 */
export function stateAt(replay: GameReplay, t: number): GameState {
	const frame = frameAt(replay, t);
	const onMark = Math.abs(t - frame.t) <= EXACT_WINDOW_S;
	const kills = replay.events.filter(
		(e): e is Extract<GameEvent, { type: 'kill' }> => e.type === 'kill' && e.t <= t
	);

	const champions = replay.participants.map((participant) => {
		const fp = frame.p.find((p) => p.id === participant.id)!;
		const pos = positionAt(replay, participant.id, t);
		let deadFor = 0;
		for (const k of kills) {
			if (k.victim !== participant.id) continue;
			const back = k.t + respawnSeconds(fp.level, k.t);
			if (t >= k.t && t < back) {
				deadFor = back - t;
			}
		}
		const lastDeathBack = kills
			.filter((k) => k.victim === participant.id)
			.map((k) => k.t + respawnSeconds(fp.level, k.t))
			.filter((b) => b <= t && b > frame.t)
			.at(-1);
		let hp = fp.hp;
		let mp = fp.mp;
		let vitalsExact = onMark;
		if (deadFor > 0) {
			hp = 0;
			vitalsExact = true;
		} else if (lastDeathBack !== undefined) {
			// Respawned since the last minute mark: full HP and mana.
			hp = fp.hpMax;
			mp = fp.mpMax;
			vitalsExact = t - lastDeathBack <= EXACT_WINDOW_S;
		}
		return {
			participant,
			x: pos.x,
			y: pos.y,
			level: fp.level,
			gold: fp.gold,
			cs: fp.cs,
			hp,
			hpMax: fp.hpMax,
			mp,
			mpMax: fp.mpMax,
			deadFor,
			exact: deadFor > 0 || pos.exact,
			vitalsExact
		};
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
