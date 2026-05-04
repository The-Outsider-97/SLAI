const TRACK = [
  [16, 84], [22, 78], [28, 72], [34, 66], [40, 60], [46, 54], [52, 48],
  [58, 42], [64, 36], [70, 30], [76, 24], [82, 18], [88, 12],
  [82, 24], [76, 30], [70, 36], [64, 42], [58, 48], [52, 54], [46, 60],
  [40, 66], [34, 72], [28, 78], [22, 84], [16, 90],
  [12, 82], [18, 76], [24, 70], [30, 64], [36, 58], [42, 52], [48, 46],
  [54, 40], [60, 34], [66, 28], [72, 22], [78, 16], [84, 10],
  [90, 18], [84, 24], [78, 30], [72, 36], [66, 42], [60, 48], [54, 54],
  [48, 60], [42, 66], [36, 72], [30, 78], [24, 84], [18, 90], [12, 88]
];

const SAFE_SPACES = new Set([0, 13, 26, 39]);
const TOKEN_COUNT = 6;

const state = {
  turn: 'red',
  red: { score: 0, tokens: Array.from({ length: TOKEN_COUNT }, (_, id) => ({ id, steps: -1 })) },
  blue: { score: 0, tokens: Array.from({ length: TOKEN_COUNT }, (_, id) => ({ id, steps: -1 })) },
  pendingRoll: null,
  winner: null,
  lastAction: '—'
};

const refs = {
  turnName: document.getElementById('turn-name'),
  winnerBanner: document.getElementById('winner-banner'),
  redCard: document.getElementById('red-card'),
  blueCard: document.getElementById('blue-card'),
  redScore: document.getElementById('red-score'),
  blueScore: document.getElementById('blue-score'),
  redLap: document.getElementById('red-lap'),
  blueLap: document.getElementById('blue-lap'),
  redHome: document.getElementById('red-home'),
  blueHome: document.getElementById('blue-home'),
  rollBtn: document.getElementById('roll-btn'),
  moveBtn: document.getElementById('move-btn'),
  lastRoll: document.getElementById('last-roll'),
  beanPips: document.querySelectorAll('#bean-pips span'),
  log: document.getElementById('log'),
  scoreLead: document.getElementById('score-lead'),
  scoreGap: document.getElementById('score-gap'),
  totalLaps: document.getElementById('total-laps'),
  scoreLastAction: document.getElementById('score-last-action'),
  bgMusic: document.getElementById('bg-music'),
  muteBtn: document.getElementById('mute-btn'),
  volumeSlider: document.getElementById('volume-slider'),
  volumeValue: document.getElementById('volume-value'),
  muteIcon: document.getElementById('mute-icon'),
  scoreSidebar: document.getElementById('scoreboard-sidebar'),
  sidebarToggle: document.getElementById('sidebar-toggle'),
  tokenLayer: document.getElementById('token-layer')
};

const startOffset = { red: 0, blue: Math.floor(TRACK.length / 2) };
const homeAnchor = { red: [10, 92], blue: [90, 8] };

function initSidebar({ container, toggleButton }) {
  if (!container || !toggleButton) return;

  const setCollapsed = (collapsed) => {
    container.classList.toggle('collapsed', collapsed);
    toggleButton.setAttribute('aria-expanded', String(!collapsed));
    toggleButton.textContent = collapsed ? '▸ Show Scoreboard' : '▸ Hide Scoreboard';
  };

  setCollapsed(false);
  toggleButton.addEventListener('click', () => {
    setCollapsed(!container.classList.contains('collapsed'));
  });
}

function appendLog(message) {
  const entry = document.createElement('p');
  entry.textContent = message;
  refs.log.prepend(entry);
  state.lastAction = message;
}

function setBeanPips(value = 0) {
  refs.beanPips.forEach((pip, index) => {
    pip.classList.toggle('active', index < Math.min(value, 5));
  });
}

function getBoardIndex(color, token) {
  if (token.steps < 0 || token.steps >= TRACK.length) return null;
  return (startOffset[color] + token.steps) % TRACK.length;
}

function renderTokens() {
  refs.tokenLayer.innerHTML = '';
  ['red', 'blue'].forEach((color) => {
    const occupancy = new Map();
    const homeStacks = [];

    state[color].tokens.forEach((token) => {
      const boardIndex = getBoardIndex(color, token);
      if (boardIndex === null) {
        homeStacks.push(token);
        return;
      }

      const stack = occupancy.get(boardIndex) || 0;
      occupancy.set(boardIndex, stack + 1);

      const [x, y] = TRACK[boardIndex];
      const tokenNode = document.createElement('div');
      tokenNode.className = `runner ${color}`;
      tokenNode.style.left = `${x + (stack % 2 ? 1.4 : -1.4)}%`;
      tokenNode.style.top = `${y + (stack > 1 ? 1.4 : 0)}%`;
      tokenNode.title = `${color.toUpperCase()} token ${token.id + 1}`;
      refs.tokenLayer.append(tokenNode);
    });

    homeStacks.forEach((token, index) => {
      const [homeX, homeY] = homeAnchor[color];
      const tokenNode = document.createElement('div');
      tokenNode.className = `runner ${color} offboard`;
      tokenNode.style.left = `${homeX + (index % 3) * 2.2}%`;
      tokenNode.style.top = `${homeY - Math.floor(index / 3) * 2.2}%`;
      tokenNode.title = `${color.toUpperCase()} token ${token.id + 1} (off board)`;
      refs.tokenLayer.append(tokenNode);
    });
  });
}

function findMovableToken(color, roll) {
  return state[color].tokens.find((token) => {
    if (token.steps === -1) return roll >= 1 && roll <= TRACK.length;
    if (token.steps >= TRACK.length) return false;
    return token.steps + roll <= TRACK.length;
  });
}

function captureAt(boardIndex, moverColor) {
  if (SAFE_SPACES.has(boardIndex)) return false;
  const enemyColor = moverColor === 'red' ? 'blue' : 'red';
  const enemy = state[enemyColor].tokens.find((token) => getBoardIndex(enemyColor, token) === boardIndex);
  if (!enemy) return false;

  enemy.steps = -1;
  state[moverColor].score += 2;
  appendLog(`${moverColor.toUpperCase()} captured ${enemyColor.toUpperCase()} on space ${boardIndex + 1} (+2).`);
  return true;
}

function syncScoreboard() {
  const scoreDiff = state.red.score - state.blue.score;
  refs.scoreGap.textContent = String(Math.abs(scoreDiff));
  refs.totalLaps.textContent = String(state.red.tokens.filter((t) => t.steps >= TRACK.length).length + state.blue.tokens.filter((t) => t.steps >= TRACK.length).length);
  refs.scoreLastAction.textContent = state.lastAction;
  refs.scoreLead.textContent = scoreDiff === 0 ? 'Tied' : scoreDiff > 0 ? 'Red + Lead' : 'Blue + Lead';
}

function syncUi() {
  refs.turnName.textContent = state.turn.toUpperCase();
  refs.turnName.className = state.turn === 'red' ? 'red-turn' : 'blue-turn';
  refs.redCard.classList.toggle('active', state.turn === 'red');
  refs.blueCard.classList.toggle('active', state.turn === 'blue');

  const redFinished = state.red.tokens.filter((t) => t.steps >= TRACK.length).length;
  const blueFinished = state.blue.tokens.filter((t) => t.steps >= TRACK.length).length;

  refs.redScore.textContent = String(state.red.score);
  refs.blueScore.textContent = String(state.blue.score);
  refs.redLap.textContent = String(redFinished);
  refs.blueLap.textContent = String(blueFinished);
  refs.redHome.textContent = String(state.red.tokens.filter((t) => t.steps === -1).length);
  refs.blueHome.textContent = String(state.blue.tokens.filter((t) => t.steps === -1).length);

  if (state.winner) {
    refs.winnerBanner.textContent = `${state.winner.toUpperCase()} COMPLETED ALL 6 TOKENS`;
    refs.winnerBanner.classList.remove('hidden');
  }

  syncScoreboard();
  renderTokens();
}

function nextTurn(extraTurn = false) {
  state.pendingRoll = null;
  refs.moveBtn.disabled = true;
  setBeanPips(0);
  if (!extraTurn) {
    state.turn = state.turn === 'red' ? 'blue' : 'red';
  }
  syncUi();
}

function completeGame() {
  refs.rollBtn.disabled = true;
  refs.moveBtn.disabled = true;
  state.winner = state.turn;
}

function applyRoll() {
  if (state.pendingRoll === null || state.winner) return;

  const moverColor = state.turn;
  const mover = state[moverColor];
  const token = findMovableToken(moverColor, state.pendingRoll);

  if (!token) {
    appendLog(`${moverColor.toUpperCase()} has no legal move for ${state.pendingRoll}.`);
    nextTurn();
    return;
  }

  if (token.steps === -1) {
    token.steps = state.pendingRoll - 1;
    appendLog(`${moverColor.toUpperCase()} entered token ${token.id + 1} and advanced ${state.pendingRoll}.`);
  } else {
    token.steps += state.pendingRoll;
    appendLog(`${moverColor.toUpperCase()} advanced token ${token.id + 1} by ${state.pendingRoll}.`);
  }

  if (token.steps === TRACK.length) {
    mover.score += 5;
    appendLog(`${moverColor.toUpperCase()} brought token ${token.id + 1} home (+5).`);
  } else {
    const boardIndex = getBoardIndex(moverColor, token);
    captureAt(boardIndex, moverColor);
  }

  if (mover.tokens.every((piece) => piece.steps >= TRACK.length)) {
    appendLog(`${moverColor.toUpperCase()} wins the match!`);
    completeGame();
    syncUi();
    return;
  }

  const extraTurn = state.pendingRoll === 5 || state.pendingRoll === 10;
  if (extraTurn) {
    appendLog(`${moverColor.toUpperCase()} earned an extra cast.`);
  }
  nextTurn(extraTurn);
}

function syncAudioUi() {
  refs.volumeValue.textContent = `${Math.round(refs.bgMusic.volume * 100)}%`;
  refs.muteIcon.src = refs.bgMusic.muted ? '../img/mute.svg' : '../img/sound_on.svg';
  refs.muteBtn.setAttribute('aria-pressed', String(refs.bgMusic.muted));
}

function initializeAudio() {
  refs.bgMusic.volume = 0.5;
  refs.bgMusic.load();
  syncAudioUi();

  const tryPlay = () => refs.bgMusic.play().catch(() => {});
  document.addEventListener('pointerdown', tryPlay, { once: true });
  document.addEventListener('keydown', tryPlay, { once: true });

  refs.muteBtn.addEventListener('click', () => {
    refs.bgMusic.muted = !refs.bgMusic.muted;
    if (!refs.bgMusic.muted) refs.bgMusic.play().catch(() => {});
    syncAudioUi();
  });

  refs.volumeSlider.addEventListener('input', (event) => {
    const volume = Number(event.target.value) / 100;
    refs.bgMusic.volume = volume;
    if (volume > 0 && refs.bgMusic.muted) refs.bgMusic.muted = false;
    syncAudioUi();
  });
}

refs.rollBtn.addEventListener('click', () => {
  if (state.pendingRoll !== null || state.winner) return;
  const rawRoll = Math.floor(Math.random() * 6);
  state.pendingRoll = rawRoll === 0 ? 10 : rawRoll;
  refs.lastRoll.textContent = `Cast: ${state.pendingRoll}`;
  setBeanPips(rawRoll);
  refs.moveBtn.disabled = false;
  appendLog(`${state.turn.toUpperCase()} cast ${state.pendingRoll} from five beans.`);
  syncUi();
});

refs.moveBtn.addEventListener('click', applyRoll);

initializeAudio();
initSidebar({ container: refs.scoreSidebar, toggleButton: refs.sidebarToggle });
setBeanPips(0);
syncUi();
appendLog('Match started. Each side must take all 6 tokens across the X-board track.');
syncUi();