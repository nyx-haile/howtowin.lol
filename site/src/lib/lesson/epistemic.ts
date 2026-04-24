/**
 * Epistemic hedging for pedagogy copy.
 *
 * The lit review (docs/site_pedagogy_engagement_lit_review.md §Risks) warns
 * that a counterfactual engine that asserts certainty ("this ward would have
 * saved you") loses user trust the first time it is wrong. All user-facing
 * claims about cohort outcomes route through this helper so the strength of
 * the hedge tracks the strength of the evidence (cohort entropy).
 */

export type Confidence = 'strong' | 'moderate' | 'weak';

/** cohort_entropy lives in [0, 1] (binary entropy over blue-win label). */
export function confidenceFromEntropy(cohortEntropy: number): Confidence {
	if (cohortEntropy < 0.3) return 'strong';
	if (cohortEntropy < 0.7) return 'moderate';
	return 'weak';
}

/**
 * Wrap a raw claim in language that matches our confidence in it.
 *
 * Input copy should be written as a bare assertion ("the fight goes your
 * way"). The helper prefixes the hedge. Fixture authors and the
 * decision-diff pipeline (bead 9b01bf83-8hi) both feed raw claims; neither
 * should hand-hedge because then the hedge doesn't track entropy.
 */
export function hedge(claim: string, confidence: Confidence): string {
	const trimmed = claim.trim().replace(/^[A-Z]/, (c) => c.toLowerCase());
	switch (confidence) {
		case 'strong':
			return `In most cohort games, ${trimmed}.`;
		case 'moderate':
			return `Cohort evidence suggests ${trimmed}.`;
		case 'weak':
			return `This signal is noisy, but ${trimmed}.`;
	}
}

/** Human-readable cohort size language for captions. */
export function cohortSizeLabel(size: number): string {
	if (size >= 64) return `${size} similar games`;
	if (size >= 16) return `${size} near-neighbors`;
	return `${size} cohort games`;
}

/** Format a cohort winrate as "X of Y won" language, avoiding bare percentages
 *  which read as stronger than the evidence supports. */
export function cohortOutcomeLabel(winrate: number, size: number): string {
	const wins = Math.round(winrate * size);
	return `${wins} of ${size} cohort games ended in a win`;
}
