import { writable, type Readable } from 'svelte/store';
import { browser } from '$app/environment';

const KEY_COUNT = 'hw:streak:count';
const KEY_LAST = 'hw:streak:lastMeaningfulISO';
const KEY_COMMITS = 'hw:streak:commits';
const RING_MAX = 30;

export interface StreakCommit {
	intent: string;
	iso: string;
}

export interface StreakState {
	count: number;
	lastMeaningfulISO: string | null;
	commits: StreakCommit[];
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

function loadInitial(): StreakState {
	return {
		count: readLocal<number>(KEY_COUNT, 0),
		lastMeaningfulISO: readLocal<string | null>(KEY_LAST, null),
		commits: readLocal<StreakCommit[]>(KEY_COMMITS, []),
	};
}

function daysBetweenISO(a: string, b: string): number {
	const ms = new Date(b).getTime() - new Date(a).getTime();
	return Math.floor(ms / (1000 * 60 * 60 * 24));
}

function createStreakStore() {
	const store = writable<StreakState>(loadInitial());

	function recordIntervention(intent: string): void {
		const trimmed = intent.trim();
		if (!trimmed) return;
		const nowISO = new Date().toISOString();
		store.update((s) => {
			let nextCount = s.count;
			if (s.lastMeaningfulISO === null) {
				nextCount = 1;
			} else {
				const gap = daysBetweenISO(s.lastMeaningfulISO, nowISO);
				if (gap === 0) {
					nextCount = Math.max(s.count, 1);
				} else if (gap === 1) {
					nextCount = s.count + 1;
				} else {
					nextCount = 1;
				}
			}
			const nextCommits = [{ intent: trimmed, iso: nowISO }, ...s.commits].slice(0, RING_MAX);
			const next: StreakState = {
				count: nextCount,
				lastMeaningfulISO: nowISO,
				commits: nextCommits,
			};
			writeLocal(KEY_COUNT, next.count);
			writeLocal(KEY_LAST, next.lastMeaningfulISO);
			writeLocal(KEY_COMMITS, next.commits);
			return next;
		});
	}

	function reset(): void {
		const cleared: StreakState = { count: 0, lastMeaningfulISO: null, commits: [] };
		writeLocal(KEY_COUNT, cleared.count);
		writeLocal(KEY_LAST, cleared.lastMeaningfulISO);
		writeLocal(KEY_COMMITS, cleared.commits);
		store.set(cleared);
	}

	const readable: Readable<StreakState> = { subscribe: store.subscribe };

	return {
		subscribe: readable.subscribe,
		recordIntervention,
		reset,
	};
}

export const streakStore = createStreakStore();
export type StreakStore = typeof streakStore;
