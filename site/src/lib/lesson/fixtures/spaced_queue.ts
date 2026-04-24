/** Stub data for Layer 4 — SpacedQueueStrip. Structure is load-bearing for the
 *  interleaving principle (lit review §4.1): items alternate themes so the
 *  queue is visibly not blocked-by-topic. */
export interface SpacedQueueItem {
	id: string;
	dayOffset: number;
	theme: string;
	themeColorVar: string;
	summary: string;
}

export const spacedQueueItems: SpacedQueueItem[] = [
	{
		id: 'q-1',
		dayOffset: 2,
		theme: 'vision',
		themeColorVar: '--concept-vision',
		summary: 'Same warding decision, different jungler matchup.',
	},
	{
		id: 'q-2',
		dayOffset: 2,
		theme: 'tempo',
		themeColorVar: '--concept-macro',
		summary: 'Minute-13 recall trade — different game, same tempo cost.',
	},
	{
		id: 'q-3',
		dayOffset: 5,
		theme: 'vision',
		themeColorVar: '--concept-vision',
		summary: 'Deep ward before scuttle — drilled in a Lee Sin game this time.',
	},
	{
		id: 'q-4',
		dayOffset: 14,
		theme: 'tempo',
		themeColorVar: '--concept-macro',
		summary: 'Late-game structure trades when you have a small lead.',
	},
];
