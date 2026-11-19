import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CODE_DIR = resolve(__dirname, '../../../../code');
const QUERY_SCRIPT = resolve(CODE_DIR, 'query.py');
const DEFAULT_PYTHON = resolve(__dirname, '../../../../.venv/bin/python3');

/**
 * JSON stdin/stdout bridge to code/query.py.
 *
 * Only works on a Node host that has the repo checked out next to the site
 * (local dev, or a self-hosted box with the model). It does NOT run on
 * Cloudflare Pages; callers must catch the failure and degrade.
 *
 * HOWL_PYTHON overrides the interpreter (e.g. code/.venv/bin/python).
 */
export async function queryPython<T>(
	request: Record<string, unknown>,
	opts: { timeoutMs?: number } = {}
): Promise<T> {
	// Imported lazily so the Cloudflare bundle never evaluates node:child_process.
	const { execFile } = await import('node:child_process');
	const python = process.env.HOWL_PYTHON || DEFAULT_PYTHON;
	return new Promise((resolvePromise, reject) => {
		const child = execFile(
			python,
			[QUERY_SCRIPT],
			{ timeout: opts.timeoutMs ?? 30000, cwd: CODE_DIR, maxBuffer: 8 * 1024 * 1024 },
			(error, stdout, stderr) => {
				if (error) {
					reject(new Error(`Python query failed: ${stderr || error.message}`));
					return;
				}
				try {
					resolvePromise(JSON.parse(stdout) as T);
				} catch {
					reject(new Error(`Invalid JSON from Python: ${stdout.slice(0, 200)}`));
				}
			}
		);
		child.stdin?.write(JSON.stringify(request));
		child.stdin?.end();
	});
}
