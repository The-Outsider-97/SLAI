const LIGHT_LAYER_ID = 'page-background-lights';

export function mountPageBackgroundLights() {
  if (typeof document === 'undefined' || document.getElementById(LIGHT_LAYER_ID)) return;

  const lightsLayer = document.createElement('div');
  lightsLayer.id = LIGHT_LAYER_ID;
  lightsLayer.setAttribute('aria-hidden', 'true');
  lightsLayer.className = 'page-bg-lights';
  lightsLayer.innerHTML = `
    <div class="page-bg-grid"></div>
    <div class="page-bg-beam page-bg-beam--sky"></div>
    <div class="page-bg-beam page-bg-beam--red"></div>
  `;

  document.body.prepend(lightsLayer);
}
