import type { LessonFixture } from './index';

export const fixture: LessonFixture = {
	id: 'loss_prose_missing_ambiguous',
	summary:
		'Blue-side loss with a mistake anchor present but no cohort prose — exercises the graceful-degradation path for when 8hi content is absent.',
	lesson: {
		match_id: 'NA1_MOCK_0107',
		team: 'blue',
		player_won: false,
		mistake_anchor: {
			minute: 18,
			cohort_entropy: 0.68,
			cohort_winrate: 0.52,
			cohort_size: 64,
			cohort_match_ids: [
				'NA1_COHORT_0311',
				'NA1_COHORT_0322',
				'NA1_COHORT_0339',
				'NA1_COHORT_0354',
			],
			polarity: 'mistake',
		},
		strength_anchor: null,
		checkpoint_sha: 'mock-plan-b-v1-13da88f',
		index_checkpoint_sha: 'mock-m4-index-v1-b7a1c22',
		n_mid_anchors: 9,
	},
	extras: {
		cohortProse: null,
		pretestChoices: [],
		interventionSuggestions: [],
	},
};
