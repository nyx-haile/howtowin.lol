/** Champion display names and Data Dragon portraits. */
const DISPLAY: Record<string, string> = {
	MissFortune: 'Miss Fortune',
	Kaisa: "Kai'Sa",
	KSante: "K'Sante",
	Khazix: "Kha'Zix",
	Chogath: "Cho'Gath",
	Velkoz: "Vel'Koz",
	Belveth: "Bel'Veth",
	MonkeyKing: 'Wukong',
	JarvanIV: 'Jarvan IV',
	LeeSin: 'Lee Sin',
	XinZhao: 'Xin Zhao',
	TwistedFate: 'Twisted Fate',
	AurelionSol: 'Aurelion Sol',
	DrMundo: 'Dr. Mundo',
	MasterYi: 'Master Yi',
	TahmKench: 'Tahm Kench',
	RekSai: "Rek'Sai",
	KogMaw: "Kog'Maw",
	Nunu: 'Nunu & Willump',
	RenataGlasc: 'Renata Glasc'
};

export function championName(key: string): string {
	return DISPLAY[key] ?? key;
}

/** Data Dragon version for a match's "14.15.607.22" game version. */
export function ddragonVersion(patch: string): string {
	const [major, minor] = patch.split('.');
	return `${major}.${minor}.1`;
}

export function portraitUrl(key: string, patch: string): string {
	return `https://ddragon.leagueoflegends.com/cdn/${ddragonVersion(patch)}/img/champion/${key}.png`;
}

export function initials(key: string): string {
	const name = championName(key).replace(/[^A-Za-z ]/g, '');
	const parts = name.split(' ').filter(Boolean);
	return (parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2)).toUpperCase();
}
