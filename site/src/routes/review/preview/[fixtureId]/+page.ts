import { error } from '@sveltejs/kit';
import { getFixture } from '$lib/lesson/fixtures';
import type { PageLoad } from './$types';

export const prerender = false;
export const ssr = true;

export const load: PageLoad = ({ params }) => {
	const fixture = getFixture(params.fixtureId);
	if (!fixture) {
		throw error(404, `No preview fixture: ${params.fixtureId}`);
	}
	return { fixture };
};
