/**
 * Drift check: every field in schema.lock.json must appear in schema.ts.
 * Run: `bun run check:schema`
 *
 * The lock is authoritative (generated from code/model/lesson.py). If a field
 * is added or renamed on the Python side, re-run the exporter and mirror the
 * change into schema.ts — this script will fail loudly until both sides match.
 */
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const LOCK_PATH = resolve(ROOT, 'src/lib/lesson/schema.lock.json');
const TS_PATH = resolve(ROOT, 'src/lib/lesson/schema.ts');

type LockedField = { name: string; type: unknown };
type LockedClass = { name: string; fields: LockedField[] };
type Lock = { schemaVersion: number; source: string; dataclasses: LockedClass[] };

const lock = JSON.parse(readFileSync(LOCK_PATH, 'utf8')) as Lock;
const ts = readFileSync(TS_PATH, 'utf8');

const problems: string[] = [];

for (const cls of lock.dataclasses) {
	const interfaceRegex = new RegExp(
		`export\\s+interface\\s+${cls.name}\\s*\\{([\\s\\S]*?)\\n\\}`,
		'm',
	);
	const match = ts.match(interfaceRegex);
	if (!match) {
		problems.push(`interface ${cls.name} is not exported from schema.ts`);
		continue;
	}
	const body = match[1];
	for (const field of cls.fields) {
		const fieldRegex = new RegExp(`\\b${field.name}\\s*[?:]`);
		if (!fieldRegex.test(body)) {
			problems.push(`field ${cls.name}.${field.name} missing from schema.ts`);
		}
	}
}

if (problems.length > 0) {
	console.error('schema drift detected:');
	for (const p of problems) console.error(`  - ${p}`);
	console.error(
		'\nfix: re-run `uv run python -m model.export_lesson_schema` from the repo root and mirror new fields into site/src/lib/lesson/schema.ts',
	);
	process.exit(1);
}

console.log(`schema-lock check OK (${lock.dataclasses.length} interfaces, source ${lock.source})`);
