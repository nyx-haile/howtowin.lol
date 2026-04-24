/** Stub data for Layer 6 — WeeklyLeagueCard. Shape reflects the "weekly
 *  cohort of same-rank improvers" variant preferred over Duolingo's XP
 *  leaderboards (lit review §Brilliant — gated on learning activity, not
 *  streaks). */
export interface LeagueStanding {
	name: string;
	intentsThisWeek: number;
}

export const myLeague = {
	name: 'Rookie league',
	size: 30,
	myIntentsThisWeek: 3,
	weeklyTargetIntents: 5,
	topPeer: 'flashy-bombadil-9',
};

export const peerStandings: LeagueStanding[] = [
	{ name: 'flashy-bombadil-9', intentsThisWeek: 6 },
	{ name: 'you', intentsThisWeek: 3 },
	{ name: 'patient-finch-3', intentsThisWeek: 3 },
	{ name: 'vocal-juniper-7', intentsThisWeek: 1 },
];
