export const WAITLIST_ENDPOINT = '/api/waitlist';
export const WAITLIST_HONEYPOT_FIELD = 'company';

export const WAITLIST_SUCCESS_MESSAGE = "You're on the list. We'll email you when reviews open up.";
export const WAITLIST_FAILURE_MESSAGE =
	'Could not join the waitlist right now. Please try again in a bit.';

export type WaitlistFormStatus = 'idle' | 'submitting' | 'success' | 'error';

export interface WaitlistResponse {
	ok: boolean;
	message: string;
}
