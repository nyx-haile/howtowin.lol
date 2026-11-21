import { execFile } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const QUERY_SCRIPT = resolve(__dirname, '../../../../code/query.py');
const PYTHON = resolve(__dirname, '../../../../.venv/bin/python3');

export function queryPython<T>(request: Record<string, unknown>): Promise<T> {
	return new Promise((resolve, reject) => {
		const child = execFile(PYTHON, [QUERY_SCRIPT], { timeout: 30000 }, (error, stdout, stderr) => {
			if (error) {
				reject(new Error(`Python query failed: ${stderr || error.message}`));
				return;
			}
			try {
				resolve(JSON.parse(stdout) as T);
			} catch {
				reject(new Error(`Invalid JSON from Python: ${stdout.slice(0, 200)}`));
			}
		});
		child.stdin?.write(JSON.stringify(request));
		child.stdin?.end();
	});
}
