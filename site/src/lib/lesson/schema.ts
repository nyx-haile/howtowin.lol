/**
 * Lesson output types. Mirrors the Python dataclasses.
 *
 * @see code/model/lesson.py  (AnchorCandidate, LessonResult)
 *
 * Drift is guarded by site/scripts/check-lesson-schema.ts, which diffs this
 * file against site/src/lib/lesson/schema.lock.json (exported by
 * `uv run python -m model.export_lesson_schema`). When the Python dataclass
 * grows a field, re-run the exporter and mirror the field here.
 */

export type Polarity = 'mistake' | 'strength';
export type Team = 'blue' | 'red';

export interface AnchorCandidate {
	minute: number;
	cohort_entropy: number;
	cohort_winrate: number;
	cohort_size: number;
	cohort_match_ids: string[];
	polarity: Polarity;
}

export interface LessonResult {
	match_id: string;
	team: Team;
	player_won: boolean;
	mistake_anchor: AnchorCandidate | null;
	strength_anchor: AnchorCandidate | null;
	checkpoint_sha: string;
	index_checkpoint_sha: string;
	n_mid_anchors: number;
}

/**
 * Presentation-only wrapper layered over a raw LessonResult. These fields
 * exist so stakeholder copy, pretest choices, and intervention suggestions
 * live alongside the lesson without leaking back into the Python dataclass.
 * Fields beyond those mirrored from LessonResult are hand-curated here and
 * filled by either fixtures (today) or the decision-diff pipeline (bead
 * 9b01bf83-8hi, tracked by 9b01bf83-b8n).
 */
export interface LessonViewModel {
	raw: LessonResult;
	anchor: AnchorCandidate | null;
	polarity: Polarity | 'neutral';
	hasAnchor: boolean;
	cohortProse: string | null;
	pretestChoices: PretestChoice[];
	interventionSuggestions: string[];
	scaffoldLevel: 0 | 1 | 2;
}

export interface PretestChoice {
	id: string;
	label: string;
	isCohortMajority?: boolean;
	note?: string;
}
