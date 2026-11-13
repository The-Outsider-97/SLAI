import { cpSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

export function publishStartup(dist) {
  if (readFileSync(join(dist, 'index.html'), 'utf8').includes('startup/loader.js')) {
    cpSync(new URL('../web/startup/', import.meta.url), join(dist, 'startup'), { recursive: true });
  }
}
