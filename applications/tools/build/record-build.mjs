// Writes dist/build.json: the SHA-256 of every file the client build reads, so `python -m applications.<app> --check` can tell a current build from a stale one by
// content rather than by file times. Python computes the same map independently (startup.py: client_inputs); a test compares the two on the real tree.
// The files are: everything under the app's src/ and under the shared UI kit's web/src/ (skipping node_modules), plus the two TypeScript configurations.
// Keys are paths relative to applications/ with forward slashes, which is how Python writes them on every platform.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, relative, sep } from 'node:path';

function walk(dir, out) {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules') continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (entry.isFile() || (entry.isSymbolicLink() && statSync(full).isFile())) out.push(full);
  }
}

export function inputFiles(appDir) {
  const applications = dirname(appDir);
  const tools = join(applications, 'tools');
  const files = [];
  walk(join(appDir, 'src'), files);
  walk(join(tools, 'web', 'src'), files);
  for (const f of [join(appDir, 'tsconfig.build.json'), join(tools, 'tsconfig.json')]) if (existsSync(f)) files.push(f);
  const key = (f) => relative(applications, f).split(sep).join('/');
  return files.map((f) => [key(f), f]).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}

export function recordBuild(appDir, dist) {
  const files = {};
  for (const [key, f] of inputFiles(appDir)) files[key] = createHash('sha256').update(readFileSync(f)).digest('hex');
  writeFileSync(join(dist, 'build.json'), JSON.stringify({ format: 1, built: new Date().toISOString().replace(/\.\d+Z$/, '+00:00'), files }, null, 1) + '\n');
  return basename(appDir);
}
