/**
 * Compact game replay built from Riot match-v5 + timeline-v5 data.
 * See samples/*.json; produced by site/scripts/compact-riot-game.py.
 * Coordinates are Summoner's Rift game units (0..~15000, origin = blue base).
 */
export type Side = 'blue' | 'red';

export interface Participant {
	id: number;
	champion: string; // Data Dragon key, e.g. "MissFortune"
	team: Side;
	role: string; // TOP | JUNGLE | MIDDLE | BOTTOM | UTILITY
}

export interface FrameParticipant {
	id: number;
	x: number;
	y: number;
	level: number;
	gold: number;
	cs: number;
}

export interface Frame {
	t: number; // seconds
	p: FrameParticipant[];
}

export type GameEvent =
	| {
			t: number;
			type: 'kill';
			killer: number;
			victim: number;
			assists: number[];
			x: number;
			y: number;
	  }
	| {
			t: number;
			type: 'objective';
			monster: string;
			sub: string | null;
			team: Side;
			killer: number;
			x: number;
			y: number;
	  }
	| {
			t: number;
			type: 'building';
			building: string;
			tower: string | null;
			lane: string;
			owner: Side;
			killer: number;
			x: number;
			y: number;
	  };

export interface GameReplay {
	matchId: string;
	patch: string;
	durationS: number;
	winner: Side;
	participants: Participant[];
	frames: Frame[];
	events: GameEvent[];
}
