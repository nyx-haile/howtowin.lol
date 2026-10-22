/**
 * ESLint rule: no-epistemic-overclaim
 *
 * Defense-in-depth behind `hedge()` in `src/lib/lesson/epistemic.ts`.
 *
 * The lit review (docs/site_pedagogy_engagement_lit_review.md §Risks) warns that a
 * counterfactual engine which asserts certainty loses user trust the first time it is
 * wrong. `hedge()` is the positive control: raw claims go in, entropy-matched hedged
 * sentences come out. This rule is the negative control — it catches copy that was
 * written straight into a literal or a Svelte template and therefore never passed
 * through the helper.
 *
 * The banned-phrase table below is a mechanical transcription of
 * `docs/site_copy_guide.md` §"Banned phrases → replacement patterns", plus the two
 * hard "never" lines from §"Copy around streaks" and §"Push copy".
 *
 * Node coverage: Literal (plain strings), TemplateElement (the static chunks of a
 * template literal), SvelteText (raw text between tags) and SvelteLiteral (unquoted
 * or quoted attribute-value text). Comments are exempt — see `isInsideComment`.
 */

/**
 * @typedef {object} BannedPhrase
 * @property {string} id        Stable slug, used in the report message.
 * @property {RegExp} pattern   Must be sticky-free and global (`g`) so every hit reports.
 * @property {string} why       Why the guide bans it.
 * @property {string} instead   The replacement pattern the guide prescribes.
 */

/** @type {BannedPhrase[]} */
const BANNED_PHRASES = [
	{
		id: 'you-will',
		pattern: /\byou(?:'|’)?ll\b|\byou will\b/gi,
		why: 'predictive certainty we cannot back',
		instead: '"this tends to correlate with ..."'
	},
	{
		id: 'unhedged-counterfactual',
		pattern: /\bwould have\b|\bwould(?:'|’)ve\b/gi,
		why: 'unhedged counterfactual',
		instead: 'a raw claim routed through hedge() — it emits "Cohort evidence suggests ..."'
	},
	{
		id: 'should-always',
		pattern: /\bshould always\b|\bshould never\b/gi,
		why: 'prescriptive; ignores context',
		instead: '"most winners in your cohort ..."'
	},
	{
		id: 'archetype-identity',
		pattern: /\byou(?:(?:'|’)re| are)\s+an?\s+[A-Z][A-Za-z-]+/g,
		why: 'archetypes are latent (bead 9b01bf83-8hw); identity labels are not observable',
		instead: 'describe the behavior directly: "your first gank averages 5:30 vs cohort 3:45"'
	},
	{
		id: 'bare-percentage-chance',
		pattern: /\d+(?:\.\d+)?\s*%\s*chance\b|\bchance of winning\b|\bwin probability of\b/gi,
		why: 'readers promote percentages to facts',
		instead: '"X of Y cohort games ended in a win" (see cohortOutcomeLabel)'
	},
	{
		id: 'shaming-stat',
		pattern:
			/\byour\s+[A-Za-z/]+\s+(?:is|was|are|were)\s+(?:bad|terrible|awful|poor|garbage|trash)\b/gi,
		why: 'shames the player; also descriptive-not-causal',
		instead: 'name the specific decision and the moment it happened'
	},
	{
		id: 'correct-answer',
		pattern: /\bcorrect answer\b|\bwrong answer\b/gi,
		why: 'the pretest is retrieval practice, not a quiz',
		instead: '"what cohort winners usually chose"'
	},
	{
		id: 'streak-loss-aversion',
		pattern:
			/\b(?:don(?:'|’)?t|do not)\s+(?:break|lose)\s+your\s+streak\b|\bstreak\s+(?:is\s+)?at\s+risk\b/gi,
		why: 'loss-aversion pattern the lit review flags as corrosive in a coaching product',
		instead: '"you saw the lesson — want to commit an intent?"'
	}
];

/**
 * Collect every comment range in the file, so text that lives inside a comment can be
 * skipped. Two sources matter here:
 *
 *  1. `sourceCode.getAllComments()` — JS/TS `//` and block comments. These are not
 *     Literal/TemplateElement nodes, so they never reach our visitors anyway; we still
 *     range-check because a Svelte `<script>` block's comments are reported through the
 *     same source code object and the belt-and-braces check costs nothing.
 *  2. Svelte HTML comments (`<!-- ... -->`). `svelte-eslint-parser` models these as
 *     `SvelteHTMLComment` nodes rather than `SvelteText`, but older parser versions have
 *     leaked comment bodies through as text, so we walk the AST for them explicitly.
 *
 * @param {import('eslint').SourceCode} sourceCode
 * @returns {[number, number][]} sorted, non-overlapping-enough ranges
 */
function collectCommentRanges(sourceCode) {
	/** @type {[number, number][]} */
	const ranges = [];

	for (const comment of sourceCode.getAllComments()) {
		if (comment.range) ranges.push([comment.range[0], comment.range[1]]);
	}

	// Svelte HTML comments live on the template AST, not in getAllComments().
	const root = sourceCode.ast;
	const seen = new Set();
	/** @param {unknown} node */
	const walk = (node) => {
		if (!node || typeof node !== 'object') return;
		if (Array.isArray(node)) {
			for (const child of node) walk(child);
			return;
		}
		if (seen.has(node)) return;
		seen.add(node);
		const typed = /** @type {{ type?: string; range?: [number, number] }} */ (node);
		if (
			typeof typed.type === 'string' &&
			(typed.type === 'SvelteHTMLComment' || typed.type === 'SvelteComment') &&
			Array.isArray(typed.range)
		) {
			ranges.push([typed.range[0], typed.range[1]]);
		}
		for (const key of Object.keys(node)) {
			if (key === 'parent') continue;
			walk(/** @type {Record<string, unknown>} */ (node)[key]);
		}
	};
	walk(root.body);
	if (/** @type {{ html?: unknown }} */ (root).html)
		walk(/** @type {{ html: unknown }} */ (root).html);

	ranges.sort((a, b) => a[0] - b[0]);
	return ranges;
}

/**
 * @param {[number, number][]} ranges
 * @param {number} start
 * @param {number} end
 */
function isInsideComment(ranges, start, end) {
	for (const [from, to] of ranges) {
		if (from > start) break;
		if (start >= from && end <= to) return true;
	}
	return false;
}

/**
 * A string Literal is only user-facing copy in some syntactic positions. Import
 * specifiers, non-computed property keys and TS literal types are never rendered, so
 * flagging them is pure noise.
 *
 * @param {import('estree').Node & { parent?: any }} node
 */
function isNonCopyLiteral(node) {
	const parent = node.parent;
	if (!parent) return false;
	switch (parent.type) {
		case 'ImportDeclaration':
		case 'ExportNamedDeclaration':
		case 'ExportAllDeclaration':
		case 'ImportExpression':
			return parent.source === node;
		case 'Property':
			return parent.key === node && !parent.computed;
		case 'MemberExpression':
			return parent.property === node && !parent.computed;
		case 'TSLiteralType':
		case 'TSEnumMember':
		case 'TSPropertySignature':
			return true;
		default:
			return false;
	}
}

/** @type {import('eslint').Rule.RuleModule} */
const rule = {
	meta: {
		type: 'problem',
		docs: {
			description:
				'Ban epistemic overclaim in user-facing pedagogy copy; route cohort claims through hedge().',
			recommended: false,
			url: 'docs/site_copy_guide.md'
		},
		schema: [
			{
				type: 'object',
				properties: {
					/**
					 * Literal substrings that suppress the rule for the text chunk containing
					 * them. Escape hatch for prose that legitimately quotes a banned phrase
					 * outside a comment (e.g. an in-app style guide page).
					 */
					allow: {
						type: 'array',
						items: { type: 'string' },
						uniqueItems: true
					},
					/** Phrase ids from the table above to disable wholesale. */
					disablePhrases: {
						type: 'array',
						items: { type: 'string' },
						uniqueItems: true
					}
				},
				additionalProperties: false
			}
		],
		messages: {
			overclaim:
				'Epistemic overclaim "{{match}}" ({{id}}): {{why}}. Use {{instead}}. See docs/site_copy_guide.md.'
		}
	},

	create(context) {
		const options = context.options[0] ?? {};
		const allow = options.allow ?? [];
		const disabled = new Set(options.disablePhrases ?? []);
		const phrases = BANNED_PHRASES.filter((p) => !disabled.has(p.id));
		const sourceCode = context.sourceCode ?? context.getSourceCode();

		/** @type {[number, number][] | null} */
		let commentRanges = null;
		const getCommentRanges = () => {
			if (commentRanges === null) commentRanges = collectCommentRanges(sourceCode);
			return commentRanges;
		};

		/**
		 * Scan a node's *raw source text* rather than its cooked value. Raw text keeps
		 * source offsets exact, which is what lets us underline the offending phrase
		 * instead of the whole string.
		 *
		 * @param {any} node
		 */
		function check(node) {
			if (!node || !Array.isArray(node.range)) return;
			const [nodeStart, nodeEnd] = node.range;
			if (isInsideComment(getCommentRanges(), nodeStart, nodeEnd)) return;

			const text = sourceCode.getText(node);
			if (!text) return;
			if (allow.some((needle) => text.includes(needle))) return;

			for (const phrase of phrases) {
				phrase.pattern.lastIndex = 0;
				let match;
				while ((match = phrase.pattern.exec(text)) !== null) {
					// Zero-length matches would spin forever.
					if (match[0].length === 0) {
						phrase.pattern.lastIndex += 1;
						continue;
					}
					const start = nodeStart + match.index;
					const end = start + match[0].length;
					if (isInsideComment(getCommentRanges(), start, end)) continue;
					context.report({
						loc: {
							start: sourceCode.getLocFromIndex(start),
							end: sourceCode.getLocFromIndex(end)
						},
						messageId: 'overclaim',
						data: {
							match: match[0],
							id: phrase.id,
							why: phrase.why,
							instead: phrase.instead
						}
					});
				}
			}
		}

		return {
			Literal(node) {
				if (typeof node.value !== 'string') return;
				if (isNonCopyLiteral(node)) return;
				check(node);
			},
			TemplateElement(node) {
				check(node);
			},
			SvelteText(node) {
				check(node);
			},
			SvelteLiteral(node) {
				check(node);
			}
		};
	}
};

export default rule;
export { BANNED_PHRASES };
