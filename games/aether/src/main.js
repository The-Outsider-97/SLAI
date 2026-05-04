import { BOARD_SIZE, POWER_WELLS, TILE_CONNECTIONS } from './constants.js';
import { createInitialState, executeAction, getRotatedConnections } from './utils/gameLogic.js';
import { requestAetherMove } from './service/aetherAiClient.js';
import { computeMatchRecord, SCOREBOARD_STORAGE_KEY } from './utils/scoring.js';

const guideBtn = document.getElementById('guide-toggle');
const guide = document.getElementById('quick-guide');
const board = document.getElementById('board');
const sidebar = document.getElementById('right-sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const tabs = Array.from(document.querySelectorAll('.tab'));
const commsPanel = document.getElementById('comms-panel');
const scorePanel = document.getElementById('score-panel');
const modeSelect = document.getElementById('mode-select');
const turnLabel = document.getElementById('turn-label');
const actionsLabel = document.getElementById('actions-label');
const actionCards = document.getElementById('action-cards');
const logBox = document.getElementById('log-box');
const p1Res = document.getElementById('p1-resonators');
const p2Res = document.getElementById('p2-resonators');
const p1Wells = document.getElementById('p1-wells');
const p2Wells = document.getElementById('p2-wells');
const p1Card = document.getElementById('player-1');
const p2Card = document.getElementById('player-2');
const scoreMode = document.getElementById('score-mode');
const scoreTurn = document.getElementById('score-turn');
const scoreWells = document.getElementById('score-wells');
const scoreWinner = document.getElementById('score-winner');
const scoreScore = document.getElementById('score-score');
const scorePoints = document.getElementById('score-points');
const scoreHistory = document.getElementById('score-history');
const endgameOverlay = document.getElementById('endgame-overlay');
const endgameHeadline = document.getElementById('endgame-headline');
const endgameSubtitle = document.getElementById('endgame-subtitle');
const endgameWinner = document.getElementById('endgame-winner');
const restartBtn = document.getElementById('restart-btn');

let gameState = createInitialState({ mode: 'PVAI', aiStarts: true });
let activityLog = ['System ready.'];
let aiTurnToken = 0;
let aiActionLog = [];
let matchLogged = false;

const SCOREBOARD_MAX_MATCHES = 12;
const AI_MATCH_LOG_KEY = 'aether_shift_ai_match_log_v1';

const isPowerWell = (row, col) => POWER_WELLS.some((well) => well.row === row && well.col === col);
const wellOwner = (row, col) => gameState.capturedWells[`${row},${col}`] || null;

const postSystemMessage = (message) => {
  if (!message) return;
  activityLog = [message, ...activityLog].slice(0, 22);
};

const getStoredList = (storageKey) => {
  try {
    const raw = localStorage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const saveStoredList = (storageKey, record, max = SCOREBOARD_MAX_MATCHES) => {
  const history = getStoredList(storageKey);
  history.unshift(record);
  localStorage.setItem(storageKey, JSON.stringify(history.slice(0, max)));
};

const recordAiAction = (entry) => {
  aiActionLog = [
    ...aiActionLog,
    {
      turn: gameState.turn,
      timestamp: new Date().toISOString(),
      ...entry,
    },
  ];
};

const selectedAction = () => {
  if (!gameState.selectedCardId || gameState.selectedActionIndex === null) return null;
  const card = gameState.faceUpCards.find((entry) => entry.id === gameState.selectedCardId);
  return card?.actions?.[gameState.selectedActionIndex] || null;
};

const serializeConnections = (tile) => {
  if (!tile) return [false, false, false, false];
  if (tile.type in TILE_CONNECTIONS) {
    return getRotatedConnections(tile.type, tile.rotation || 0);
  }
  return [false, false, false, false];
};

const canAct = () => !gameState.winner && !(gameState.mode === 'PVAI' && gameState.activePlayer === 2);

const renderLog = () => {
  logBox.innerHTML = activityLog.map((line) => `<p>${line}</p>`).join('');
};

const renderScore = () => {
  const redWells = Object.values(gameState.capturedWells).filter((owner) => owner === 1).length;
  const blueWells = Object.values(gameState.capturedWells).filter((owner) => owner === 2).length;
  const matchRecord = computeMatchRecord(gameState);

  scoreMode.textContent = `Mode: ${gameState.mode === 'PVAI' ? 'Player v. AI' : 'Player v. Player'}`;
  scoreTurn.textContent = `Turn: ${gameState.turn}`;
  scoreWells.textContent = `Wells — Red: ${redWells} / Blue: ${blueWells}`;
  scoreWinner.textContent = `Winner: ${gameState.winner ? gameState.players[gameState.winner].name : 'In Progress'}`;
  scoreScore.textContent = `Score: ${matchRecord.score}`;
  scorePoints.textContent = `Points: ${matchRecord.points}`;

  const completedMatches = getStoredList(SCOREBOARD_STORAGE_KEY);
  if (!completedMatches.length) {
    scoreHistory.innerHTML = '<p class="empty-history">No completed matches yet.</p>';
    return;
  }

  scoreHistory.innerHTML = completedMatches.map((entry, idx) => `
    <article class="history-card">
      <p class="history-top">Match ${completedMatches.length - idx} · ${entry.timestamp}</p>
      <p>Mode: ${entry.mode}</p>
      <p>Winner: ${entry.winner}</p>
      <p>Turns/Rounds: ${entry.turns} / ${entry.rounds}</p>
      <p>Wells — Red: ${entry.wells?.red ?? 0} / Blue: ${entry.wells?.blue ?? 0}</p>
      <p>Score: ${entry.score}</p>
      <p>Points: ${entry.points}</p>
      <p>AI actions logged: ${entry.aiActions ?? 0}</p>
    </article>
  `).join('');
};

const finalizeMatchIfNeeded = () => {
  if (!gameState.winner || matchLogged) return null;

  const record = {
    ...computeMatchRecord(gameState),
    aiActions: aiActionLog.length,
  };

  saveStoredList(SCOREBOARD_STORAGE_KEY, record);
  saveStoredList(AI_MATCH_LOG_KEY, {
    id: record.id,
    timestamp: record.timestamp,
    mode: record.mode,
    winner: record.winner,
    score: record.score,
    points: record.points,
    turns: record.turns,
    rounds: record.rounds,
    reason: gameState.winReason,
    aiActions: aiActionLog,
    systemLog: activityLog,
  }, 40);

  matchLogged = true;
  return record;
};

const renderPlayerPanels = () => {
  const redWells = Object.values(gameState.capturedWells).filter((owner) => owner === 1).length;
  const blueWells = Object.values(gameState.capturedWells).filter((owner) => owner === 2).length;
  const redRemaining = gameState.players[1].resonators;
  const blueRemaining = gameState.players[2].resonators;

  p1Res.textContent = `${'● '.repeat(redRemaining)}${'○ '.repeat(4 - redRemaining)}`.trim();
  p2Res.textContent = `${'● '.repeat(blueRemaining)}${'○ '.repeat(4 - blueRemaining)}`.trim();
  p1Wells.textContent = `${'⚡ '.repeat(redWells)}${'· '.repeat(3 - redWells)}`.trim();
  p2Wells.textContent = `${'⚡ '.repeat(blueWells)}${'· '.repeat(3 - blueWells)}`.trim();

  p1Card.classList.toggle('active', gameState.activePlayer === 1);
  p2Card.classList.toggle('active', gameState.activePlayer === 2);
};

const tileGlyph = (tile) => {
  if (!tile) return '';
  if (tile.type === 'CROSS') return '╋';
  if (tile.type === 'STRAIGHT') return '┃';
  if (tile.type === 'CURVE') return '┗';
  return '┻';
};

const renderBoard = () => {
  board.innerHTML = '';
  const active = selectedAction();

  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const tileData = gameState.board[row][col];
      const tileBtn = document.createElement('button');
      tileBtn.type = 'button';
      tileBtn.className = 'tile';
      tileBtn.dataset.row = String(row);
      tileBtn.dataset.col = String(col);

      if (isPowerWell(row, col)) {
        tileBtn.classList.add('well');
      }

      if (tileData) {
        tileBtn.classList.add(`tile-${tileData.color || 'neutral'}`);
        const glyph = document.createElement('span');
        glyph.className = 'path-glyph';
        glyph.textContent = tileGlyph(tileData);
        glyph.style.transform = `rotate(${tileData.rotation || 0}deg)`;
        tileBtn.appendChild(glyph);

        if (tileData.hasResonator) {
          const resonator = document.createElement('span');
          resonator.className = `resonator r-${tileData.resonatorOwner}`;
          resonator.textContent = '⬢';
          tileBtn.appendChild(resonator);
        }
      }

      if (isPowerWell(row, col)) {
        const owner = wellOwner(row, col);
        const well = document.createElement('span');
        well.className = `well-icon ${owner ? `owner-${owner}` : ''}`;
        well.textContent = '⚡';
        tileBtn.appendChild(well);
      }

      const currentConnections = serializeConnections(tileData);
      if (currentConnections.some(Boolean)) {
        const conn = document.createElement('span');
        conn.className = 'conn';
        conn.dataset.n = String(currentConnections[0]);
        conn.dataset.e = String(currentConnections[1]);
        conn.dataset.s = String(currentConnections[2]);
        conn.dataset.w = String(currentConnections[3]);
        tileBtn.appendChild(conn);
      }

      const playersPresent = tileData?.playersPresent || [];
      if (playersPresent.length) {
        const tokenWrap = document.createElement('span');
        tokenWrap.className = 'tokens';
        playersPresent.forEach((pid) => {
          const token = document.createElement('span');
          token.className = `token token-${pid}`;
          token.textContent = '◉';
          tokenWrap.appendChild(token);
        });
        tileBtn.appendChild(tokenWrap);
      }

      if (!canAct()) tileBtn.disabled = true;

      tileBtn.addEventListener('click', () => handleTileClick(row, col, active));
      board.appendChild(tileBtn);
    }
  }
};

const renderActionCards = () => {
  actionCards.innerHTML = '';
  gameState.faceUpCards.forEach((card) => {
    card.actions.forEach((action, actionIndex) => {
      const selected = gameState.selectedCardId === card.id && gameState.selectedActionIndex === actionIndex;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `card ${selected ? 'selected' : ''}`;
      button.textContent = action;
      button.disabled = !canAct();
      button.addEventListener('click', () => {
        gameState.selectedCardId = card.id;
        gameState.selectedActionIndex = actionIndex;
        render();
      });
      actionCards.appendChild(button);
    });
  });
};

const renderHeader = () => {
  const active = gameState.players[gameState.activePlayer];
  turnLabel.textContent = gameState.winner ? `${gameState.players[gameState.winner].name} wins` : `${active.name}'s Turn`;
  turnLabel.className = gameState.activePlayer === 1 ? 'red-turn' : 'blue-turn';
  actionsLabel.textContent = `ACTIONS: ${gameState.actionsRemaining}`;
  modeSelect.value = gameState.mode;
};


const renderEndgameOverlay = () => {
  if (!gameState.winner) {
    endgameOverlay.classList.add('hidden');
    return;
  }

  const playerWon = gameState.winner === 1 || gameState.mode === 'PVP';
  endgameHeadline.textContent = playerWon ? 'VICTORY' : 'DEFEAT';
  endgameHeadline.className = gameState.winner === 1 ? 'winner-red' : 'winner-blue';
  endgameSubtitle.textContent = gameState.winReason === 'Path Completed!'
    ? 'Edge link established.'
    : 'Well dominance achieved.';
  endgameWinner.textContent = `${gameState.players[gameState.winner].name} wins the match.`;
  endgameOverlay.classList.remove('hidden');
};

const render = () => {
  finalizeMatchIfNeeded();
  renderHeader();
  renderPlayerPanels();
  renderActionCards();
  renderBoard();
  renderLog();
  renderScore();
  renderEndgameOverlay();
};

const handleTileClick = async (row, col, action) => {
  if (!canAct() || !action) return;
  const result = executeAction(gameState, action, { row, col });
  postSystemMessage(result.message);
  if (!result.success) {
    render();
    return;
  }

  gameState = result.newState;
  render();

  if (gameState.winner) {
    await notifyLearnEndpoint();
    return;
  }

  if (gameState.mode === 'PVAI' && gameState.activePlayer === 2) {
    queueAiTurn();
  }
};

const queueAiTurn = () => {
  const token = ++aiTurnToken;
  setTimeout(async () => {
    if (token !== aiTurnToken || gameState.winner || gameState.activePlayer !== 2 || gameState.mode !== 'PVAI') return;

    const move = await requestAetherMove(gameState);
    if (!move) {
      postSystemMessage('AI skipped turn.');
      recordAiAction({ type: 'SKIP' });
      gameState = { ...gameState, activePlayer: 1, actionsRemaining: 2 };
      render();
      return;
    }

    gameState = { ...gameState, selectedCardId: move.cardId, selectedActionIndex: move.actionIndex };
    render();

    const card = gameState.faceUpCards.find((entry) => entry.id === move.cardId);
    const action = card?.actions?.[move.actionIndex];
    if (!action) {
      postSystemMessage('AI selected an invalid action.');
      recordAiAction({ type: 'INVALID_MOVE', move });
      render();
      return;
    }

    recordAiAction({
      type: 'SELECTED_ACTION',
      cardId: move.cardId,
      actionIndex: move.actionIndex,
      action,
      target: move.target || null,
    });

    const result = executeAction(gameState, action, move.target);
    postSystemMessage(result.message || 'AI action resolved.');
    recordAiAction({
      type: result.success ? 'ACTION_RESOLVED' : 'ACTION_FAILED',
      action,
      target: move.target || null,
      message: result.message || null,
    });
    if (result.success) {
      gameState = result.newState;
      render();
      if (gameState.winner) {
        await notifyLearnEndpoint();
        return;
      }

      if (gameState.mode === 'PVAI' && gameState.activePlayer === 2 && gameState.actionsRemaining > 0) {
        queueAiTurn();
      }
      return;
    }

    postSystemMessage('AI action failed. Ending AI turn.');
    gameState = { ...gameState, activePlayer: 1, actionsRemaining: 2, selectedCardId: null, selectedActionIndex: null };
    render();
  }, 700);
};

const notifyLearnEndpoint = async () => {
  const matchRecord = finalizeMatchIfNeeded() || computeMatchRecord(gameState);
  try {
    await fetch('/api/ai/learn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game: 'aether_shift',
        matchRecord,
        winner: gameState.winner,
        mode: gameState.mode,
        turns: gameState.turn,
        capturedWells: gameState.capturedWells,
        reason: gameState.winReason,
        aiActions: aiActionLog,
        systemLog: activityLog,
      }),
    });
  } catch (error) {
    console.warn('Failed to post learning update', error);
  }
};

modeSelect.addEventListener('change', () => {
  const nextMode = modeSelect.value;
  const aiStarts = nextMode === 'PVAI';
  gameState = createInitialState({ mode: nextMode, aiStarts });
  aiTurnToken += 1;
  aiActionLog = [];
  matchLogged = false;
  activityLog = [`Mode switched: ${nextMode === 'PVAI' ? 'Player v. AI' : 'Player v. Player'}.`];
  render();
  if (gameState.mode === 'PVAI' && gameState.activePlayer === 2) queueAiTurn();
});

guideBtn.addEventListener('click', () => {
  const nowHidden = !guide.classList.contains('hidden');
  guide.classList.toggle('hidden');
  guideBtn.setAttribute('aria-expanded', String(!nowHidden));
});

sidebarToggle.addEventListener('click', () => {
  const collapsed = sidebar.classList.toggle('collapsed');
  sidebarToggle.textContent = collapsed ? '❯' : '❮';
});

tabs.forEach((tabButton) => {
  tabButton.addEventListener('click', () => {
    tabs.forEach((btn) => btn.classList.remove('active'));
    tabButton.classList.add('active');
    const isComms = tabButton.dataset.tab === 'comms';
    commsPanel.classList.toggle('hidden', !isComms);
    scorePanel.classList.toggle('hidden', isComms);
  });
});

render();
if (gameState.mode === 'PVAI' && gameState.activePlayer === 2) queueAiTurn();

restartBtn.addEventListener('click', () => {
  gameState = createInitialState({ mode: gameState.mode, aiStarts: gameState.mode === 'PVAI' });
  aiTurnToken += 1;
  aiActionLog = [];
  matchLogged = false;
  activityLog = ['New game initialized.'];
  render();
  if (gameState.mode === 'PVAI' && gameState.activePlayer === 2) queueAiTurn();
});