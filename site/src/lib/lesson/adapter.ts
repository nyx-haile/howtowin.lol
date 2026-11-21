import type {
	AnchorCandidate,
	LessonResult,
	LessonViewModel,
	Polarity,
} from '$lib/lesson/schema';
import { confidenceFromEntropy, hedge } from '$lib/lesson/epistemic';
import type { FixtureExtras } from '$lib/lesson/fixtures';

/**
 * Extras for a lesson that comes straight from the model with nothing
 * authored on top. Until the decision-diff pipeline (bead 9b01bf83-8hi)
 * lands, this is what every live lesson gets: no pretest, no cohort prose,
 * no suggested intents. LessonStage skips the pretest when choices are empty.
 */
export const MODEL_ONLY_EXTRAS: FixtureExtras = {
	cohortProse: null,
	pretestChoices: [],
	interventionSuggestions: [],
};

/**
 * Build the presentation view-model from a raw LessonResult plus the
 * non-Python presentation extras (pretest choices, cohort prose, intervention
 * suggestions).
 *
 * Today `extras` is a hand-authored fixture. When the decision-diff pipeline
 * (bead 9b01bf83-8hi → follow-ups 9b01bf83-dcy and 9b01bf83-b8n) lands, its
 * output flows through the same extras shape. The cohort prose goes through
 * `hedge()` here — that's the enforcement point that guarantees no unhedged
 * claim can reach the UI, whether authored or generated.
 */
export function toLessonViewModel(
	raw: LessonResult,
	extras: FixtureExtras,
): LessonViewModel {
	const anchor: AnchorCandidate | null =
		raw.mistake_anchor ?? raw.strength_anchor ?? null;

	const polarity: Polarity | 'neutral' = anchor ? anchor.polarity : 'neutral';

	const cohortProse = hedgeIfPresent(extras.cohortProse, anchor);

	return {
		raw,
		anchor,
		polarity,
		hasAnchor: anchor !== null,
		cohortProse,
		pretestChoices: extras.pretestChoices,
		interventionSuggestions: extras.interventionSuggestions,
		scaffoldLevel: 0,
	};
}

function hedgeIfPresent(
	claim: string | null,
	anchor: AnchorCandidate | null,
): string | null {
	if (!claim) return null;
	if (!anchor) return claim;
	return hedge(claim, confidenceFromEntropy(anchor.cohort_entropy));
}
