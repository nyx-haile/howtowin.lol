import { writable, type Readable } from 'svelte/store';
import { browser } from '$app/environment';

const KEY_COMPLETED = 'hw:onboarding:lessonsCompleted';
const KEY_FIRST_ISO = 'hw:onboarding:firstCompletedISO';

export interface OnboardingState {
	lessonsCompleted: number;
	firstCompletedISO: string | null;
}

function readLocal<T>(key: string, fallback: T): T {
	if (!browser) return fallback;
	try {
		const raw = localStorage.getItem(key);
		if (raw === null) return fallback;
		return JSON.parse(raw) as T;
	} catch {
		return fallback;
	}
}

function writeLocal(key: string, value: unknown): void {
	if (!browser) return;
	try {
		localStorage.setItem(key, JSON.stringify(value));
	} catch {
		// ignore quota errors
	}
}

function loadInitial(): OnboardingState {
	return {
		lessonsCompleted: readLocal<number>(KEY_COMPLETED, 0),
		firstCompletedISO: readLocal<string | null>(KEY_FIRST_ISO, null),
	};
}

function createOnboardingStore() {
	const store = writable<OnboardingState>(loadInitial());

	function recordLessonCompleted(): void {
		const nowISO = new Date().toISOString();
		store.update((s) => {
			const next: OnboardingState = {
				lessonsCompleted: s.lessonsCompleted + 1,
				firstCompletedISO: s.firstCompletedISO ?? nowISO,
			};
			writeLocal(KEY_COMPLETED, next.lessonsCompleted);
			writeLocal(KEY_FIRST_ISO, next.firstCompletedISO);
			return next;
		});
	}

	function reset(): void {
		const cleared: OnboardingState = { lessonsCompleted: 0, firstCompletedISO: null };
		writeLocal(KEY_COMPLETED, cleared.lessonsCompleted);
		writeLocal(KEY_FIRST_ISO, cleared.firstCompletedISO);
		store.set(cleared);
	}

	const readable: Readable<OnboardingState> = { subscribe: store.subscribe };

	return {
		subscribe: readable.subscribe,
		recordLessonCompleted,
		reset,
	};
}

export const onboardingStore = createOnboardingStore();
export type OnboardingStore = typeof onboardingStore;
