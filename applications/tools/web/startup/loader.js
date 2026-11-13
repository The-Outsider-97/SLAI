// A classic script placed immediately before the application's module entry point.
(() => {
  const { name, logo, fallback } = document.currentScript.dataset;
  const panel = document.createElement('dialog');
  panel.className = 'slai-startup';
  panel.setAttribute('aria-label', name + ' startup');
  const orb = document.createElement('div');
  orb.className = 'slai-startup-orb';
  orb.setAttribute('aria-hidden', 'true');
  const mark = new Image();
  mark.alt = '';
  mark.addEventListener('error', () => {
    if (fallback && mark.getAttribute('src') !== fallback) mark.src = fallback;
    else mark.remove();
  });
  mark.src = logo;
  orb.append(mark);
  const message = document.createElement('p');
  message.setAttribute('role', 'status');
  message.textContent = 'Loading ' + name;
  panel.append(orb, message);
  document.body.append(panel);
  panel.showModal();

  const listeners = new AbortController();
  const options = { signal: listeners.signal };
  panel.addEventListener('cancel', event => event.preventDefault(), options);
  const timeout = setTimeout(() => {
    message.textContent = name + ' is taking longer than expected.';
    controls();
  }, 45000);

  function dismiss() {
    clearTimeout(timeout);
    listeners.abort();
    panel.close();
    panel.remove();
  }
  function controls() {
    if (panel.querySelector('button')) return;
    const retry = document.createElement('button');
    retry.textContent = 'Reload';
    retry.onclick = () => location.reload();
    const close = document.createElement('button');
    close.textContent = 'Show application';
    close.onclick = dismiss;
    panel.append(retry, close);
  }
  function fail(detail) {
    clearTimeout(timeout);
    if (panel.classList.contains('failed')) return;
    panel.classList.add('failed');
    message.setAttribute('role', 'alert');
    message.textContent = detail || (name + ' could not start. Check the launch window for details.');
    controls();
  }
  window.addEventListener('slai:startup-update', event => {
    if (panel.classList.contains('failed')) return;
    const { message: status, progress } = event.detail || {};
    if (typeof status === 'string' && status.trim()) message.textContent = status;
    if (!Number.isFinite(progress) || progress < 0 || progress > 1) return;
    let meter = panel.querySelector('progress');
    if (!meter) {
      meter = document.createElement('progress');
      meter.max = 1;
      meter.setAttribute('aria-label', name + ' startup progress');
      panel.append(meter);
    }
    meter.value = progress;
  }, options);
  window.addEventListener('slai:ready', () => {
    if (!panel.classList.contains('failed')) dismiss();
  }, options);
  window.addEventListener('slai:startup-error', event => fail(event.detail), options);
  // Only the application's own code can fail its start-up. Browser extensions (wallets, translators, ad blockers) throw on the same page, and so do
  // other sites' frames; none of that is ours, and it must neither stop the animation nor block the ready signal.
  const ours = address => {
    try {
      const url = new URL(address, window.location.href);
      return url.origin === window.location.origin && /^\/(js|startup|tools)\//.test(url.pathname);
    } catch { return false; }
  };
  window.addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement) {
      if (ours(event.target.src)) fail();
    } else if (event instanceof ErrorEvent && ours(event.filename)) fail(event.message);
  }, { ...options, capture: true });
  window.addEventListener('unhandledrejection', event => {
    const reason = event.reason;
    if (typeof reason?.stack === 'string' && reason.stack.includes(window.location.origin + '/js/')) fail(reason.message);
  }, options);
})();
