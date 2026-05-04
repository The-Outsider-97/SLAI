import { createBoard, renderBoard, renderThrow, throwSticks, TOKEN_COUNT, TRACK_LENGTH, SAFE_CELLS } from './board.js';
import { typeWriter } from './typeWriter.js';

// Game state
const state = {
  turn: 'light',
  pendingDistance: 0,
  winner: null,
  light: {
    tokens: Array.from({ length: TOKEN_COUNT }, (_, id) => ({ id, position: -1, finished: false }))
  },
  dark: {
    tokens: Array.from({ length: TOKEN_COUNT }, (_, id) => ({ id, position: -1, finished: false }))
  },
};

// DOM references
const refs = {
  board: document.getElementById('board'),
  throwTray: document.getElementById('throw-tray'),
  throwBtn: document.getElementById('throw-btn'),
  moveBtn: document.getElementById('move-btn'),
  turnName: document.getElementById('turn-name'),
  log: document.getElementById('log'),
  lightCard: document.getElementById('light-card'),
  darkCard: document.getElementById('dark-card'),
  bgMusic: document.getElementById('bg-music'),
  muteBtn: document.getElementById('mute-btn'),
  volumeSlider: document.getElementById('volume-slider'),
  volumeValue: document.getElementById('volume-value'),
  muteIcon: document.getElementById('mute-icon'),
  winnerBanner: document.getElementById('winner-banner'),
  modePvp: document.getElementById('mode-pvp'),
  modePvai: document.getElementById('mode-pvai'),
  newGameBtn: document.getElementById('new-game-btn'),
  quoteText: document.getElementById('quote-text'),
  quoteMeta: document.getElementById('quote-meta')
};

// Stats display elements
const statRefs = {
  light: {
    onboard: document.getElementById('light-onboard'),
    reserve: document.getElementById('light-reserve'),
    finished: document.getElementById('light-finished'),
  },
  dark: {
    onboard: document.getElementById('dark-onboard'),
    reserve: document.getElementById('dark-reserve'),
    finished: document.getElementById('dark-finished'),
  },
};

let selectedTokenId = null;
const cells = createBoard(refs.board, handleCellClick);
let gameMode = 'pvp';
let isAITurn = false;

// Quotes state
let facts =[];
let cancelQuoteType = null;
let cancelMetaType = null;

// ========== Quotes fetching and updating ==========
async function loadFacts() {
  try {
    const response = await fetch('../templates/facts.json');
    if (response.ok) {
      const data = await response.json();
      facts = data.facts ||[];
      showRandomQuote();
      // Update quote every 5 minutes (300,000 ms)
      setInterval(showRandomQuote, 300000);
    }
  } catch (err) {
    console.error("Failed to load facts", err);
  }
}

function showRandomQuote() {
  if (!facts.length) return;
  const fact = facts[Math.floor(Math.random() * facts.length)];

  if (cancelQuoteType) cancelQuoteType();
  if (cancelMetaType) cancelMetaType();
  
  refs.quoteMeta.textContent = ''; // Clear meta text

  const quoteStr = `"${fact.description}"`;
  const metaStr = `${fact.category} | ${fact.title} | ${fact.source.name}: ${fact.source.reference}`;

  cancelQuoteType = typeWriter(refs.quoteText, quoteStr, 30, () => {
    cancelMetaType = typeWriter(refs.quoteMeta, metaStr, 25);
  });
}

// ========== Helper functions ==========
function activeTokens(color) {
  return state[color].tokens.filter((token) => token.position >= 0 && !token.finished).length;
}

function reserveTokens(color) {
  return state[color].tokens.filter((token) => token.position < 0 && !token.finished).length;
}

function finishedTokens(color) {
  return state[color].tokens.filter((token) => token.finished).length;
}

function updatePanels() {
  for (const color of['light', 'dark']) {
    statRefs[color].onboard.textContent = String(activeTokens(color));
    statRefs[color].reserve.textContent = String(reserveTokens(color));
    statRefs[color].finished.textContent = String(finishedTokens(color));
  }

  const isLight = state.turn === 'light';
  refs.turnName.textContent = state.turn.toUpperCase();
  refs.turnName.style.color = isLight ? '#e3c9a0' : '#b98f67';
  refs.lightCard.classList.toggle('active', isLight);
  refs.darkCard.classList.toggle('active', !isLight);
}

function log(message) {
  const line = document.createElement('p');
  line.textContent = message;
  refs.log.prepend(line);
}

function getFirstMovableToken(color) {
  return state[color].tokens.find((token) => !token.finished) ?? null;
}

function findTokenAt(color, position) {
  return state[color].tokens.find((token) => token.position === position && !token.finished);
}

function endTurn() {
  selectedTokenId = null;
  state.pendingDistance = 0;
  refs.moveBtn.disabled = true;
  state.turn = state.turn === 'light' ? 'dark' : 'light';
  updatePanels();

  if (!state.winner && gameMode === 'pvai' && state.turn === 'dark') {
    setTimeout(makeAIMove, 400);
  }
}

function checkWinner() {
  for (const color of['light', 'dark']) {
    if (finishedTokens(color) === TOKEN_COUNT) {
      state.winner = color;
      refs.winnerBanner.classList.remove('hidden');
      refs.winnerBanner.textContent = `${color.toUpperCase()} wins Puluc!`;
      refs.throwBtn.disabled = true;
      refs.moveBtn.disabled = true;
      log(`${color.toUpperCase()} has escorted all warriors off the path.`);
      return true;
    }
  }
  return false;
}

function applyCapture(color, position) {
  const enemy = color === 'light' ? 'dark' : 'light';
  if (SAFE_CELLS.has(position)) return;

  const enemyToken = findTokenAt(enemy, position);
  if (!enemyToken) return;

  enemyToken.position = -1;
  const alliedStack = state[color].tokens.filter((token) => token.position === position && !token.finished).length;
  log(`${color.toUpperCase()} captures on ${position + 1}. ${enemy.toUpperCase()} warrior returned to start.`);
  if (alliedStack > 1) {
    log(`${color.toUpperCase()} now controls a stack of ${alliedStack} warriors on that cell.`);
  }
}

function moveSelectedToken() {
  if (state.pendingDistance <= 0 || state.winner) return;

  const color = state.turn;
  const token = state[color].tokens.find((item) => item.id === selectedTokenId) ?? getFirstMovableToken(color);
  if (!token) {
    endTurn();
    return;
  }

  const origin = token.position;
  const current = token.position < 0 ? -1 : token.position;
  const next = current + state.pendingDistance;

  if (next >= TRACK_LENGTH) {
    token.finished = true;
    token.position = TRACK_LENGTH;
    log(`${color.toUpperCase()} advances ${state.pendingDistance} and exits the board.`);
  } else {
    token.position = next;
    applyCapture(color, next);
    const fromText = origin < 0 ? 'start' : String(origin + 1);
    log(`${color.toUpperCase()} moves from ${fromText} to ${next + 1}.`);
  }

  renderBoard(cells, state);
  updatePanels();
  if (!checkWinner()) {
    endTurn();
  }
}

function handleCellClick(index) {
  if (state.pendingDistance <= 0 || state.winner) return;
  if (gameMode === 'pvai' && state.turn === 'dark') return;

  const token = findTokenAt(state.turn, index);
  selectedTokenId = token?.id ?? null;
}

// ========== AI move (simple random) ==========
function makeAIMove() {
  if (state.winner || isAITurn || state.turn !== 'dark') return;
  isAITurn = true;

  const result = throwSticks(4);
  state.pendingDistance = result.moveDistance;
  renderThrow(refs.throwTray, result.sticks);
  log(`DARK AI throws sticks: move ${result.moveDistance}.`);

  const color = 'dark';
  const availableTokens = state[color].tokens.filter(t => !t.finished);
  if (availableTokens.length === 0) {
    endTurn();
    isAITurn = false;
    return;
  }
  const randomToken = availableTokens[Math.floor(Math.random() * availableTokens.length)];
  selectedTokenId = randomToken.id;

  setTimeout(() => {
    moveSelectedToken();
    isAITurn = false;
    if (!state.winner && gameMode === 'pvai' && state.turn === 'dark') {
      setTimeout(makeAIMove, 600);
    }
  }, 500);
}

// ========== Reset game ==========
function resetGame() {
  state.turn = 'light';
  state.pendingDistance = 0;
  state.winner = null;
  state.light.tokens = Array.from({ length: TOKEN_COUNT }, (_, id) => ({ id, position: -1, finished: false }));
  state.dark.tokens = Array.from({ length: TOKEN_COUNT }, (_, id) => ({ id, position: -1, finished: false }));
  selectedTokenId = null;
  isAITurn = false;

  renderBoard(cells, state);
  updatePanels();
  refs.throwBtn.disabled = false;
  refs.moveBtn.disabled = true;
  refs.winnerBanner.classList.add('hidden');
  refs.throwTray.innerHTML = '';
  refs.log.innerHTML = '';
  log('New game started. Throw sticks to begin.');
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

// ========== Event listeners ==========
refs.throwBtn.addEventListener('click', () => {
  if (state.winner) return;
  if (gameMode === 'pvai' && state.turn === 'dark') return;

  const result = throwSticks(4);
  state.pendingDistance = result.moveDistance;
  renderThrow(refs.throwTray, result.sticks);
  refs.moveBtn.disabled = false;
  log(`${state.turn.toUpperCase()} throws sticks: move ${result.moveDistance}.`);
});

refs.moveBtn.addEventListener('click', moveSelectedToken);

refs.modePvp.addEventListener('click', () => {
  gameMode = 'pvp';
  refs.modePvp.classList.add('active');
  refs.modePvai.classList.remove('active');
  resetGame();
});
refs.modePvai.addEventListener('click', () => {
  gameMode = 'pvai';
  refs.modePvai.classList.add('active');
  refs.modePvp.classList.remove('active');
  resetGame();
});

refs.newGameBtn.addEventListener('click', resetGame);

// Initial render
initializeAudio();
loadFacts();
renderBoard(cells, state);
updatePanels();
log('Puluc has begun. Throw sticks to move your first warrior.');