import type { LessonFixture } from './index';

export const fixture: LessonFixture = {
	id: 'nothing_to_teach_early_stomp',
	summary:
		'Early stomp — no mid-game frame crossed the entropy threshold, so both anchors are null. Exercises the no-anchor presentation path.',
	lesson: {
		match_id: 'NA1_MOCK_0099',
		team: 'blue',
		player_won: true,
		mistake_anchor: null,
		strength_anchor: null,
		checkpoint_sha: 'mock-plan-b-v1-13da88f',
		index_checkpoint_sha: 'mock-m4-index-v1-b7a1c22',
		n_mid_anchors: 3,
	},
	extras: {
		cohortProse: null,
		pretestChoices: [],
		interventionSuggestions: [],
	},
};
