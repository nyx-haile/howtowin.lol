import { WAITLIST_HONEYPOT_FIELD } from '$lib/waitlist';

const WAITLIST_TABLE_SQL = `
	CREATE TABLE IF NOT EXISTS waitlist_signups (
		email TEXT PRIMARY KEY,
		source TEXT NOT NULL,
		created_at TEXT NOT NULL,
		updated_at TEXT NOT NULL,
		submissions INTEGER NOT NULL DEFAULT 1
	)
`;

const WAITLIST_UPSERT_SQL = `
	INSERT INTO waitlist_signups (email, source, created_at, updated_at, submissions)
	VALUES (?1, ?2, ?3, ?3, 1)
	ON CONFLICT(email) DO UPDATE SET
		source = excluded.source,
		updated_at = excluded.updated_at,
		submissions = waitlist_signups.submissions + 1
`;

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export interface WaitlistSignup {
	email: string;
	source: string;
}

export interface ParsedWaitlistSubmission {
	honeypotHit: boolean;
	signup?: WaitlistSignup;
	error?: string;
	status?: number;
}

function getString(value: FormDataEntryValue | null): string {
	return typeof value === 'string' ? value.trim() : '';
}

function normalizeSource(value: string): string {
	const normalized = value
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9_-]+/g, '-');

	if (!normalized) {
		return 'unknown';
	}

	return normalized.slice(0, 64);
}

export function parseWaitlistSubmission(formData: FormData): ParsedWaitlistSubmission {
	if (getString(formData.get(WAITLIST_HONEYPOT_FIELD))) {
		return { honeypotHit: true };
	}

	const email = getString(formData.get('email')).toLowerCase();

	if (!email) {
		return {
			honeypotHit: false,
			error: 'Please enter your email address.',
			status: 400
		};
	}

	if (email.length > 254 || !EMAIL_REGEX.test(email)) {
		return {
			honeypotHit: false,
			error: 'Please enter a valid email address.',
			status: 400
		};
	}

	return {
		honeypotHit: false,
		signup: {
			email,
			source: normalizeSource(getString(formData.get('source')))
		}
	};
}

export async function saveWaitlistSignup(db: D1Database, signup: WaitlistSignup): Promise<void> {
	const timestamp = new Date().toISOString();

	await db.batch([
		db.prepare(WAITLIST_TABLE_SQL),
		db.prepare(WAITLIST_UPSERT_SQL).bind(signup.email, signup.source, timestamp)
	]);
}
