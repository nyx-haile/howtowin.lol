import type { Side } from './types';

/** Summoner's Rift geometry in game units. */
export const RIFT_SIZE = 15000;

export interface TowerSpot {
	id: string;
	team: Side;
	lane: 'TOP_LANE' | 'MID_LANE' | 'BOT_LANE' | 'NEXUS';
	tier: 'OUTER_TURRET' | 'INNER_TURRET' | 'BASE_TURRET' | 'NEXUS_TURRET';
	x: number;
	y: number;
}

export const TOWERS: TowerSpot[] = [
	{ id: 'b-top-1', team: 'blue', lane: 'TOP_LANE', tier: 'OUTER_TURRET', x: 981, y: 10441 },
	{ id: 'b-top-2', team: 'blue', lane: 'TOP_LANE', tier: 'INNER_TURRET', x: 1512, y: 6699 },
	{ id: 'b-top-3', team: 'blue', lane: 'TOP_LANE', tier: 'BASE_TURRET', x: 1169, y: 4287 },
	{ id: 'b-mid-1', team: 'blue', lane: 'MID_LANE', tier: 'OUTER_TURRET', x: 5846, y: 6396 },
	{ id: 'b-mid-2', team: 'blue', lane: 'MID_LANE', tier: 'INNER_TURRET', x: 5048, y: 4812 },
	{ id: 'b-mid-3', team: 'blue', lane: 'MID_LANE', tier: 'BASE_TURRET', x: 3651, y: 3696 },
	{ id: 'b-bot-1', team: 'blue', lane: 'BOT_LANE', tier: 'OUTER_TURRET', x: 10504, y: 1029 },
	{ id: 'b-bot-2', team: 'blue', lane: 'BOT_LANE', tier: 'INNER_TURRET', x: 6919, y: 1483 },
	{ id: 'b-bot-3', team: 'blue', lane: 'BOT_LANE', tier: 'BASE_TURRET', x: 4281, y: 1253 },
	{ id: 'r-top-1', team: 'red', lane: 'TOP_LANE', tier: 'OUTER_TURRET', x: 4318, y: 13875 },
	{ id: 'r-top-2', team: 'red', lane: 'TOP_LANE', tier: 'INNER_TURRET', x: 7943, y: 13411 },
	{ id: 'r-top-3', team: 'red', lane: 'TOP_LANE', tier: 'BASE_TURRET', x: 10481, y: 13650 },
	{ id: 'r-mid-1', team: 'red', lane: 'MID_LANE', tier: 'OUTER_TURRET', x: 8955, y: 8510 },
	{ id: 'r-mid-2', team: 'red', lane: 'MID_LANE', tier: 'INNER_TURRET', x: 9767, y: 10113 },
	{ id: 'r-mid-3', team: 'red', lane: 'MID_LANE', tier: 'BASE_TURRET', x: 11134, y: 11207 },
	{ id: 'r-bot-1', team: 'red', lane: 'BOT_LANE', tier: 'OUTER_TURRET', x: 13866, y: 4505 },
	{ id: 'r-bot-2', team: 'red', lane: 'BOT_LANE', tier: 'INNER_TURRET', x: 13327, y: 8226 },
	{ id: 'r-bot-3', team: 'red', lane: 'BOT_LANE', tier: 'BASE_TURRET', x: 13624, y: 10572 }
];

export const PITS = {
	baron: { x: 5007, y: 10471, label: 'Baron / Herald' },
	dragon: { x: 9866, y: 4414, label: 'Dragon' }
};

/**
 * Respawn time by champion level (base, before the late-game time
 * multiplier). Values from the LoL wiki death-timer table; callers should
 * present results as approximate.
 */
const BASE_RESPAWN_S = [
	10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5
];

export function respawnSeconds(level: number, gameTimeS: number): number {
	const base = BASE_RESPAWN_S[Math.max(1, Math.min(18, level)) - 1];
	// Time-impact factor kicks in after 15:00 (roughly +0.425% per 30s, capped).
	const minutesPast15 = Math.max(0, gameTimeS / 60 - 15);
	const factor = Math.min(0.5, Math.ceil(minutesPast15 * 2) * 0.00425);
	return base * (1 + factor);
}

/** Rough walk time between two map points, for "who gets there first" copy. */
export const TYPICAL_MOVE_SPEED = 380;

export function walkSeconds(a: { x: number; y: number }, b: { x: number; y: number }): number {
	return Math.hypot(a.x - b.x, a.y - b.y) / TYPICAL_MOVE_SPEED;
}
