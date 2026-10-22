import js from '@eslint/js';
import ts from 'typescript-eslint';
import svelte from 'eslint-plugin-svelte';
import prettier from 'eslint-config-prettier';
import globals from 'globals';
import noEpistemicOverclaim from './eslint-rules/no-epistemic-overclaim.js';

/**
 * Local plugin holding repo-specific copy rules. Kept inline (rather than published)
 * because the banned-phrase table is a transcription of docs/site_copy_guide.md and
 * has no meaning outside this repo.
 */
const epistemicPlugin = {
	rules: {
		'no-epistemic-overclaim': noEpistemicOverclaim
	}
};

/** @type {import('eslint').Linter.Config[]} */
export default [
	js.configs.recommended,
	...ts.configs.recommended,
	...svelte.configs['flat/recommended'],
	prettier,
	...svelte.configs['flat/prettier'],
	{
		languageOptions: {
			globals: {
				...globals.browser,
				...globals.node
			}
		}
	},
	{
		files: ['**/*.svelte'],
		languageOptions: {
			parserOptions: {
				parser: ts.parser
			}
		}
	},
	// Epistemic copy gate. Scoped deliberately narrowly: only the pedagogy surfaces
	// named in docs/site_copy_guide.md. The rest of the site (marketing, admin, dev
	// tooling) is not held to the hedge rule and must not start failing lint because
	// of it.
	{
		files: ['src/lib/lesson/**/*.{js,ts,svelte}', 'src/routes/review/**/*.{js,ts,svelte}'],
		plugins: {
			epistemic: epistemicPlugin
		},
		rules: {
			'epistemic/no-epistemic-overclaim': 'error'
		}
	},
	{
		ignores: ['build/', '.svelte-kit/', 'dist/']
	}
];
