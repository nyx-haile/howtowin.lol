import { json } from '@sveltejs/kit';

import {
	WAITLIST_FAILURE_MESSAGE,
	WAITLIST_SUCCESS_MESSAGE,
	type WaitlistResponse
} from '$lib/waitlist';
import { parseWaitlistSubmission, saveWaitlistSignup } from '$lib/server/waitlist';

import type { RequestHandler } from './$types';

function response(body: WaitlistResponse, status = 200) {
	return json(body, {
		status,
		headers: {
			'cache-control': 'no-store'
		}
	});
}

export const POST: RequestHandler = async ({ platform, request }) => {
	const db = platform?.env.WAITLIST_DB;

	if (!db) {
		return response(
			{
				ok: false,
				message: WAITLIST_FAILURE_MESSAGE
			},
			503
		);
	}

	let formData: FormData;

	try {
		formData = await request.formData();
	} catch {
		return response(
			{
				ok: false,
				message: 'Invalid submission.'
			},
			400
		);
	}

	const parsed = parseWaitlistSubmission(formData);

	if (parsed.honeypotHit) {
		return response({
			ok: true,
			message: WAITLIST_SUCCESS_MESSAGE
		});
	}

	if (!parsed.signup) {
		return response(
			{
				ok: false,
				message: parsed.error ?? WAITLIST_FAILURE_MESSAGE
			},
			parsed.status ?? 400
		);
	}

	try {
		await saveWaitlistSignup(db, parsed.signup);

		return response({
			ok: true,
			message: WAITLIST_SUCCESS_MESSAGE
		});
	} catch (error) {
		console.error('Failed to save waitlist signup', error);

		return response(
			{
				ok: false,
				message: WAITLIST_FAILURE_MESSAGE
			},
			500
		);
	}
};
