import type { LessonFixture } from './index';

export const fixture: LessonFixture = {
	id: 'win_late_strength_22m',
	summary:
		'Red-side win with a strength anchor at minute 22 — positive-feedback variant where cohort entropy was high and the player took the minority-winning branch.',
	lesson: {
		match_id: 'NA1_MOCK_0042',
		team: 'red',
		player_won: true,
		mistake_anchor: null,
		strength_anchor: {
			minute: 22,
			cohort_entropy: 0.72,
			cohort_winrate: 0.41,
			cohort_size: 64,
			cohort_match_ids: [
				'NA1_COHORT_0209',
				'NA1_COHORT_0218',
				'NA1_COHORT_0221',
				'NA1_COHORT_0244',
				'NA1_COHORT_0253',
				'NA1_COHORT_0260',
				'NA1_COHORT_0271',
				'NA1_COHORT_0285',
			],
			polarity: 'strength',
		},
		checkpoint_sha: 'mock-plan-b-v1-13da88f',
		index_checkpoint_sha: 'mock-m4-index-v1-b7a1c22',
		n_mid_anchors: 15,
	},
	extras: {
		cohortProse:
			'the late-game lead holds because you trade baron for two inhibitor towers while the enemy is split',
		pretestChoices: [
			{ id: 'baron-now', label: 'baron now — we have numbers' },
			{ id: 'trade-for-inhibs', label: 'skip baron, hammer two inhib towers' },
			{ id: 'reset-and-wait', label: 'reset, wait for elder spawn' },
			{ id: 'pick-and-engage', label: 'look for a pick, force fight' },
		],
		interventionSuggestions: [
			'Before my next game: when I have a 4k+ gold lead past minute 20, I will ask "can I take two structures" before "can I take baron".',
		],
	},
};
