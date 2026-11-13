// Optional integration runner. The driver passes the launch URL and cookie via stdin.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { Window } from 'happy-dom';
import { createCanvas } from '@napi-rs/canvas';

const [app, scenario] = process.argv.slice(2);
const { url, cookie } = JSON.parse(readFileSync(0, 'utf8'));
const dist = resolve('applications', app, 'dist');
const browser = new Window({
  url,
  settings: { disableCSSFileLoading: true, disableJavaScriptFileLoading: true, disableJavaScriptEvaluation: true }
});
const { document } = browser;
document.write(readFileSync(resolve(dist, 'index.html'), 'utf8'));
const loader = document.querySelector('script[data-logo]');
Object.defineProperty(document, 'currentScript', { value: loader, configurable: true });

// Supply the DOM globals used by the unmodified compiled clients.
for (const key of [
  'window', 'document', 'Image', 'Event', 'ErrorEvent', 'CustomEvent', 'HTMLScriptElement',
  'HTMLElement', 'Element', 'Node', 'HTMLInputElement', 'HTMLTextAreaElement', 'HTMLButtonElement',
  'HTMLSelectElement', 'HTMLCanvasElement', 'DocumentFragment', 'DOMParser', 'SVGElement',
  'localStorage', 'sessionStorage', 'location', 'history', 'navigator', 'matchMedia', 'getComputedStyle',
  'requestAnimationFrame', 'ResizeObserver', 'MutationObserver', 'AbortController', 'CSS', 'CSSStyleSheet', 'DOMRect', 'getSelection'
]) {
  if (browser[key] !== undefined) Object.defineProperty(globalThis, key, { value: browser[key], configurable: true });
}
globalThis.Option = function Option(text, value) {
  const option = document.createElement('option');
  option.text = text;
  option.value = value;
  return option;
};
const canvases = new WeakMap();
browser.HTMLCanvasElement.prototype.getContext = function(kind) {
  if (kind !== '2d') return null;
  if (!canvases.has(this)) canvases.set(this, createCanvas(this.width || 300, this.height || 150));
  return canvases.get(this).getContext('2d');
};

const nativeFetch = globalThis.fetch;
const gate = Promise.withResolvers();
globalThis.fetch = async (input, options = {}) => {
  await gate.promise;
  if (scenario === 'failure') throw new Error('Service unavailable');
  const target = new URL(input, url);
  assert.equal(target.origin, new URL(url).origin);
  const headers = new Headers(options.headers);
  headers.set('Cookie', cookie);
  return nativeFetch(target, { ...options, signal: undefined, headers });
};
const terminal = Promise.withResolvers();
for (const event of ['slai:ready', 'slai:startup-error']) {
  window.addEventListener(event, () => terminal.resolve(event), { once: true });
}
await import(pathToFileURL(resolve(dist, 'startup/loader.js')));
const panel = document.querySelector('dialog.slai-startup');
assert(panel.open);
assert.equal(panel.querySelector('p').textContent, 'Loading ' + loader.dataset.name);
assert.equal(panel.querySelector('progress'), null);
window.dispatchEvent(new CustomEvent('slai:startup-update', { detail: { message: 'Reading project', progress: 0.5 } }));
assert.equal(panel.querySelector('progress').value, 0.5);
assert.equal(panel.querySelector('p').textContent, 'Reading project');
const modulePath = document.querySelector('script[type=module]').getAttribute('src').replace(/^\//, '');
await import(pathToFileURL(resolve(dist, modulePath)));
assert(panel.open);
gate.resolve();
const deadline = setTimeout(() => terminal.reject(new Error('Startup timed out')), 20000);
const event = await terminal.promise;
clearTimeout(deadline);
if (scenario === 'failure') {
  assert.equal(event, 'slai:startup-error');
  assert(panel.querySelector('[role=alert]'));
  window.dispatchEvent(new Event('slai:ready'));
  assert(panel.open);
  panel.querySelectorAll('button')[1].click();
} else {
  assert.equal(event, 'slai:ready');
}
window.dispatchEvent(new Event('slai:ready'));
assert.equal(document.querySelector('dialog.slai-startup'), null);
assert(document.querySelector('#app,#root').textContent.trim());
process.exit(0); // Hosted clients own recurring timers until their browser tab closes.
