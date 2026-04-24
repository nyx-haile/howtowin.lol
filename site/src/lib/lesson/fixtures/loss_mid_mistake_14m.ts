import type { LessonFixture } from './index';

export const fixture: LessonFixture = {
	id: 'loss_mid_mistake_14m',
	summary:
		'Blue-side loss with a high-entropy mistake anchor at minute 14 — main happy path for the mistake presentation.',
	lesson: {
		match_id: 'NA1_MOCK_0001',
		team: 'blue',
		player_won: false,
		mistake_anchor: {
			minute: 14,
			cohort_entropy: 0.85,
			cohort_winrate: 0.58,
			cohort_size: 64,
			cohort_match_ids: [
				'NA1_COHORT_0012',
				'NA1_COHORT_0019',
				'NA1_COHORT_0031',
				'NA1_COHORT_0044',
				'NA1_COHORT_0058',
				'NA1_COHORT_0067',
				'NA1_COHORT_0073',
				'NA1_COHORT_0088',
			],
			polarity: 'mistake',
		},
		strength_anchor: null,
		checkpoint_sha: 'mock-plan-b-v1-13da88f',
		index_checkpoint_sha: 'mock-m4-index-v1-b7a1c22',
		n_mid_anchors: 12,
	},
	extras: {
		cohortProse:
			'the fight unlocks because blue takes vision on the enemy jungle raptors before committing to scuttle',
		pretestChoices: [
			{ id: 'ward-deep', label: 'ward enemy raptors before approaching scuttle' },
			{
				id: 'scuttle-first',
				label: 'contest scuttle blind — we have level advantage',
			},
			{ id: 'back-and-reset', label: 'back, buy, reset mid wave' },
			{ id: 'rotate-bot', label: 'rotate bot for drake setup' },
		],
		interventionSuggestions: [
			'Before my next game: I will only contest an objective after placing a deep ward on the approach.',
			'Before my next game: I will treat minute 13–16 as a vision-first window, not a damage-first one.',
		],
	},
};
