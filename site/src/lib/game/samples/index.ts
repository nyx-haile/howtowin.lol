import type { GameReplay } from '../types';
import type { CoachLesson } from '../lesson';
import game5081314144 from './NA1_5081314144.json';
import { PITS } from '../rift';

/**
 * Sample lesson on a real ranked game (NA1_5081314144, patch 14.15) from
 * example/. The game data is real; the question and the comparison-game
 * counts are hand-written stand-ins until the model produces them.
 */
export const sampleReplays: Record<string, GameReplay> = {
	NA1_5081314144: game5081314144 as GameReplay
};

export const sampleLessons: CoachLesson[] = [
	{
		matchId: 'NA1_5081314144',
		team: 'red',
		t: 960,
		playForwardS: 105,
		question:
			'You just took their mid tower and you are up 870 gold. Where should your team be in the next 30 seconds?',
		targets: [
			{ id: 'herald', label: 'Rift Herald', x: PITS.baron.x, y: PITS.baron.y },
			{ id: 'mid-inner', label: 'Their mid tower', x: 5048, y: 4812 },
			{ id: 'top-lane', label: 'Top lane', x: 2400, y: 12900 },
			{ id: 'reset', label: 'Reset', x: 12300, y: 12300 }
		],
		actualTargetId: null,
		actualSummary:
			'Your team split up. Miss Fortune and Nidalee kept pushing mid, Gangplank stayed top, Ezreal was bot and Leona was walking back through your jungle.',
		whatHappened: [
			'16:38 Taliyah takes Rift Herald alone. Gangplank, Leona and Ezreal arrive a second late and kill her in the pit.',
			'16:40 Ezreal is caught in river and Miss Fortune dies mid. Your minions take their mid inner tower.',
			'16:50 to 17:11 the fight moves to top lane. You trade 3 kills for 5, then lose your top tower at 17:23.'
		],
		comparison: {
			games: 64,
			byTarget: [
				{ targetId: 'herald', games: 29, won: 21 },
				{ targetId: 'mid-inner', games: 14, won: 8 },
				{ targetId: 'top-lane', games: 13, won: 4 },
				{ targetId: 'reset', games: 8, won: 4 }
			]
		},
		illustrative: true
	}
];

export function getLesson(matchId: string, team: string): CoachLesson | undefined {
	return sampleLessons.find((l) => l.matchId === matchId && l.team === team);
}
