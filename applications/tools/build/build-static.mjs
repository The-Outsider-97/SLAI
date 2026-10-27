// Build an SLAI browser client. Run it from the application folder (or pass the folder):
//
//   node ../tools/build/build-static.mjs            # from applications/<app>
//   node applications/tools/build/build-static.mjs applications/<app>
//
// It compiles the app's TypeScript with the app's tsconfig.build.json into dist/js, then copies src/client/index.html and every
// src/client/*.css next to it. Uses the project's own TypeScript when `npm install` has been run, otherwise a `tsc` on the PATH.
// Icons and sounds are never copied: the server serves them from the application's assets folder and applications/assets.
// The shared UI kit (applications/tools/web) is compiled together with the app, because the app imports it by relative path
// (tsconfig.build.json sets rootDir to applications/), and its stylesheet is served by the server at /tools/slai-ui.css.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? process.cwd());
const dist = join(root, 'dist');
const client = join(root, 'src', 'client');
const config = join(root, 'tsconfig.build.json');

for (const needed of [config, join(client, 'index.html')]) {
  if (!existsSync(needed)) { console.error(`Not an SLAI application folder: ${needed} is missing.`); process.exit(2); }
}

rmSync(join(dist, 'js'), { recursive: true, force: true });       // so a renamed or deleted module can never linger in dist
mkdirSync(dist, { recursive: true });

let tsc = null;
try { tsc = createRequire(join(root, 'package.json')).resolve('typescript/bin/tsc'); } catch { /* not installed locally */ }
const args = ['-p', config];
if (tsc) execFileSync(process.execPath, [tsc, ...args], { stdio: 'inherit', cwd: root });
else execFileSync(process.platform === 'win32' ? 'tsc.cmd' : 'tsc', args, { stdio: 'inherit', cwd: root, shell: process.platform === 'win32' });

const copied = [];
for (const name of readdirSync(client)) {
  if (name === 'index.html' || name.endsWith('.css')) { cpSync(join(client, name), join(dist, name)); copied.push(name); }
}
console.log(`Built dist/ (${copied.join(', ')}, js/)`);
