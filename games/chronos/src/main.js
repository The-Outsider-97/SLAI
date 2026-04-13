import { Game } from './engine.js';
import { CONFIG, ActionType, BOARD_VARIANTS, applyBoardSize } from './constants.js';
import { GoogleGenAI } from "@google/genai";
import { mountPageBackgroundLights } from './pageBackgroundLights.js';

let game;
let selectedUnit = null;
let selectedToken = null;
let possibleActions = null;

const boardEl = document.getElementById('board');
const timelineEl = document.getElementById('timeline-slots');
const logsEl = document.getElementById('logs');
const turnIndicatorEl = document.getElementById('turn-indicator');
const phaseIndicatorEl = document.getElementById('phase-indicator');
const tokenContainerEl = document.getElementById('token-container');
const actionButtonsEl = document.getElementById('action-buttons');
const sidePanelTitleEl = document.getElementById('side-panel-title');
const commsTabBtn = document.getElementById('comms-tab-btn');
const scoreboardTabBtn = document.getElementById('scoreboard-tab-btn');
const commsPanelEl = document.getElementById('comms-panel');
const scoreboardPanelEl = document.getElementById('scoreboard-panel');
const scoreboardContentEl = document.getElementById('scoreboard-content');
const rightSidebarEl = document.getElementById('right-sidebar');
const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
const boardSizeSelectEl = document.getElementById('board-size-select');
const topCoordsEl = document.getElementById('board-top-coords');
const bottomCoordsEl = document.getElementById('board-bottom-coords');
const leftCoordsEl = document.getElementById('board-left-coords');
const rightCoordsEl = document.getElementById('board-right-coords');

const SCOREBOARD_STORAGE_KEY = 'chronos_scoreboard_history_v1';
const SCOREBOARD_MAX_MATCHES = 200;
let activeSidePanel = 'comms';
let sidePanelHidden = false;
let aiStrategosSubmitted = false;

mountPageBackgroundLights();

function initGame(boardSize = CONFIG.board.size) {
  applyBoardSize(boardSize);
  game = new Game();
  boardEl.innerHTML = '';
  selectedUnit = null;
  selectedToken = null;
  possibleActions = null;
  renderBoardCoordinates();
  render();
  log(`Game Started on ${CONFIG.board.size}x${CONFIG.board.size}. Player 1's Turn.`);
}


function renderBoardCoordinates() {
  if (!topCoordsEl || !bottomCoordsEl || !leftCoordsEl || !rightCoordsEl) return;

  const size = CONFIG.board.size;
  const letters = Array.from({ length: size }, (_, idx) => String.fromCharCode(65 + idx));

  topCoordsEl.innerHTML = letters.map(label => `<div class="flex-1 text-center">${label}</div>`).join('');
  bottomCoordsEl.innerHTML = letters.map(label => `<div class="flex-1 text-center">${label}</div>`).join('');

  leftCoordsEl.innerHTML = Array.from({ length: size }, (_, idx) => `<div class="flex-1 flex items-center justify-center">${size - idx}</div>`).join('');
  rightCoordsEl.innerHTML = Array.from({ length: size }, (_, idx) => `<div class="flex-1 flex items-center justify-center">${size - idx}</div>`).join('');
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function computeMatchScoreBreakdown({ p1Points, p2Points, turnCount, winnerLabel, p1PiecesLeft = 0, p2PiecesLeft = 0, centerControl = null }) {
  const rawPointDiff = (p1Points || 0) - (p2Points || 0);
  const pointDiffContribution = clamp(rawPointDiff * 8, -40, 40);

  const roundsUsed = Number.isFinite(turnCount) ? turnCount : 12;
  const roundContribution = clamp((12 - roundsUsed) * 2, -20, 20);
  const outcomeContribution = winnerLabel === 'Player 1' ? 40 : winnerLabel === 'Player 2' ? -40 : 0;

  const pieceDiff = (p1PiecesLeft || 0) - (p2PiecesLeft || 0);
  const pieceContribution = clamp(pieceDiff * 4, -24, 24);

  const centerContribution = centerControl?.playerId === 0
    ? centerControl.scoreWeight
    : centerControl?.playerId === 1
      ? -centerControl.scoreWeight
      : 0;

  const total = clamp(
    pointDiffContribution + roundContribution + outcomeContribution + pieceContribution + centerContribution,
    -100,
    100
  );

  return {
    rawPointDiff,
    finalScore: total,
    contributions: {
      pointDiffContribution,
      roundContribution,
      outcomeContribution,
      pieceContribution,
      centerContribution
    }
  };
}

function getCenterControl(gameRef) {
  const center = Math.floor(CONFIG.board.size / 2);
  const centerUnit = gameRef.board.getUnitAt(center, center);
  if (!centerUnit || centerUnit.health <= 0) {
    return {
      playerId: null,
      pieceType: null,
      scoreWeight: 0,
      label: 'None'
    };
  }

  return {
    playerId: centerUnit.playerId,
    pieceType: centerUnit.type,
    scoreWeight: centerUnit.props.value * 6,
    label: `Player ${centerUnit.playerId + 1} (${centerUnit.type})`
  };
}

function buildScoreTooltip(entry) {
  const piecesLine = `${entry.p1PiecesLeft ?? 0}-${entry.p2PiecesLeft ?? 0} = ${(entry.p1PiecesLeft ?? 0) - (entry.p2PiecesLeft ?? 0)}`;
  return [
    `Point Diff = ${entry.p1Points} - ${entry.p2Points} = ${entry.pointDifference}`,
    `Final Score (P1) = clamp(-100..100, PointDiffTerm + RoundsTerm + OutcomeTerm + PieceTerm + CenterTerm)`,
    `PointDiffTerm: (${entry.pointDifference}) * 8 => ${entry.scoreBreakdown?.pointDiffContribution ?? 0}`,
    `RoundsTerm: (12 - ${entry.turnCount ?? 0}) * 2 => ${entry.scoreBreakdown?.roundContribution ?? 0}`,
    `OutcomeTerm: ${entry.winner} => ${entry.scoreBreakdown?.outcomeContribution ?? 0}`,
    `PieceTerm: (${piecesLine}) * 4 => ${entry.scoreBreakdown?.pieceContribution ?? 0}`,
    `CenterTerm: ${entry.centerCoreOccupant || 'None'} => ${entry.scoreBreakdown?.centerContribution ?? 0}`
  ].join('\n');
}

function computeFinalScore(args) {
  return computeMatchScoreBreakdown(args).finalScore;
}

function computeAiReward(finalScore) {
  // AI controls Player 2, so positive reward means Player 2 favorable result.
  return Math.max(-1, Math.min(1, (-finalScore) / 100));
}

function setSidebarVisibility(hidden) {
  sidePanelHidden = hidden;
  if (!rightSidebarEl || !sidebarToggleBtn) return;

  rightSidebarEl.classList.toggle('w-80', !hidden);
  rightSidebarEl.classList.toggle('w-16', hidden);

  const collapsible = rightSidebarEl.querySelectorAll('#comms-panel, #scoreboard-panel, #side-panel-title, #comms-tab-btn, #scoreboard-tab-btn');
  collapsible.forEach((el) => {
    el.classList.toggle('hidden', hidden);
  });

  sidebarToggleBtn.textContent = hidden ? 'Show' : 'Hide';
}

function getMatchHistory() {
  try {
    const stored = localStorage.getItem(SCOREBOARD_STORAGE_KEY);
    const parsed = stored ? JSON.parse(stored) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveMatchRecord(record) {
  const history = getMatchHistory();
  history.unshift(record);
  localStorage.setItem(SCOREBOARD_STORAGE_KEY, JSON.stringify(history.slice(0, SCOREBOARD_MAX_MATCHES)));
}

function renderScoreboardPanel() {
  if (!scoreboardContentEl) return;
  const history = getMatchHistory();

  if (!history.length) {
    scoreboardContentEl.innerHTML = '<div class="text-slate-500 italic text-center">No completed matches yet.</div>';
    return;
  }

  scoreboardContentEl.innerHTML = `
    <div class="flex flex-col gap-3">
      ${history.map((entry, idx) => {
        const winnerClass = entry.winner === 'Player 1'
          ? 'text-sky-blue'
          : entry.winner === 'Player 2'
            ? 'text-traffic-red'
            : 'text-yellow-400';
        const displayFinalScore = Number.isFinite(entry.finalScore)
          ? entry.finalScore
          : computeFinalScore({
              p1Points: Number(entry.p1Points) || 0,
              p2Points: Number(entry.p2Points) || 0,
              turnCount: Number(entry.turnCount) || 0,
              winnerLabel: entry.winner || 'Draw',
              p1PiecesLeft: Number(entry.p1PiecesLeft) || 0,
              p2PiecesLeft: Number(entry.p2PiecesLeft) || 0,
              centerControl: {
                playerId: entry.centerCoreOwner ?? null,
                scoreWeight: Number(entry.centerCoreWeight) || 0
              }
            });
        const tooltip = buildScoreTooltip({
          ...entry,
          finalScore: displayFinalScore,
          pointDifference: Number(entry.pointDifference) || 0,
          scoreBreakdown: entry.scoreBreakdown || computeMatchScoreBreakdown({
            p1Points: Number(entry.p1Points) || 0,
            p2Points: Number(entry.p2Points) || 0,
            turnCount: Number(entry.turnCount) || 0,
            winnerLabel: entry.winner || 'Draw',
            p1PiecesLeft: Number(entry.p1PiecesLeft) || 0,
            p2PiecesLeft: Number(entry.p2PiecesLeft) || 0,
            centerControl: {
              playerId: entry.centerCoreOwner ?? null,
              scoreWeight: Number(entry.centerCoreWeight) || 0
            }
          }).contributions
        });
        return `
          <div class="rounded-xl border border-white/10 bg-black/30 p-3 flex flex-col gap-2">
            <div class="flex items-center justify-between">
              <div class="text-[10px] tracking-widest uppercase text-slate-500">Match ${history.length - idx}</div>
              <div class="text-[10px] text-slate-600">${entry.timestamp}</div>
            </div>
            <div class="text-[11px]"><span class="text-slate-500">Winner:</span> <span class="font-bold ${winnerClass}">${entry.winner}</span></div>
            <div class="text-[11px]"><span class="text-slate-500">Points (P1/P2):</span> <span class="font-mono text-white">${entry.p1Points} / ${entry.p2Points}</span></div>
            <div class="text-[11px]" title="${tooltip}"><span class="text-slate-500">Point Diff:</span> <span class="font-mono text-white">${entry.pointDifference}</span></div>
            <div class="text-[11px]"><span class="text-slate-500">Rounds:</span> <span class="font-mono text-white">${entry.turnCount ?? '-'}</span></div>
            <div class="text-[11px]" title="${tooltip}"><span class="text-slate-500">Final Score (P1):</span> <span class="font-mono ${displayFinalScore >= 0 ? 'text-sky-blue' : 'text-traffic-red'}">${displayFinalScore}</span></div>
            <div class="text-[11px]"><span class="text-slate-500">Pieces Left (P1/P2):</span> <span class="font-mono text-white">${entry.p1PiecesLeft ?? 0} / ${entry.p2PiecesLeft ?? 0}</span></div>
            <div class="text-[11px]"><span class="text-slate-500">Center Core Occupant:</span> <span class="text-white">${entry.centerCoreOccupant || 'None'}</span></div>
            <div class="text-[11px]"><span class="text-slate-500">Center Core Captured:</span> <span class="text-white">${entry.centerCoreCaptured ? 'Yes' : 'No'}</span></div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function switchSidePanel(panelName) {
  activeSidePanel = panelName;
  const showingComms = panelName === 'comms';

  if (sidePanelTitleEl) sidePanelTitleEl.textContent = showingComms ? 'Comms Link' : 'Scoreboard';
  if (commsPanelEl) commsPanelEl.classList.toggle('hidden', !showingComms);
  if (scoreboardPanelEl) scoreboardPanelEl.classList.toggle('hidden', showingComms);

  if (commsTabBtn) {
    commsTabBtn.className = `px-3 py-1 rounded-lg text-[10px] font-bold tracking-widest uppercase ${showingComms ? 'bg-sky-blue text-white' : 'text-slate-400'}`;
  }

  if (scoreboardTabBtn) {
    scoreboardTabBtn.className = `px-3 py-1 rounded-lg text-[10px] font-bold tracking-widest uppercase ${showingComms ? 'text-slate-400' : 'bg-sky-blue text-white'}`;
  }

  if (!showingComms) {
    renderScoreboardPanel();
  }
}

function log(msg) {
  const div = document.createElement('div');
  div.textContent = msg;
  div.className = 'mb-1 text-slate-400 text-sm font-mono';
  logsEl.prepend(div);
}

function render() {
  // Reset AI strategos flag if we are no longer in that phase
  if (game.phase !== 'strategos_decision') {
    aiStrategosSubmitted = false;
  }

  renderBoard();
  renderTimeline();
  renderInfo();
  updateControls();

  if (game.phase === 'game_over') {
    if (!document.getElementById('game-over-modal') && !window.gameOverTimer) {
      // Delay showing the modal to let the user see the final state (winning move)
      window.gameOverTimer = setTimeout(() => {
        renderGameOver();
        window.gameOverTimer = null;
      }, 2500);
    }
  } else {
    const modal = document.getElementById('game-over-modal');
    if (modal) modal.remove();
    if (window.gameOverTimer) {
      clearTimeout(window.gameOverTimer);
      window.gameOverTimer = null;
    }
  }

  // If we just entered strategos decision, let the AI choose once
  if (game.phase === 'strategos_decision' && !aiStrategosSubmitted) {
    aiStrategosChoice();
  }
}

async function aiStrategosChoice() {
    aiStrategosSubmitted = true;
    try {
        const state = game.serialize();
        const response = await fetch('/api/ai/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state)
        });
        const data = await response.json();
        if (data.choice) {
            game.submitMutualStrategosChoice(1, data.choice);
            log(`AI chose ${data.choice}.`);
            render();
        }
    } catch (e) {
        console.error("AI strategos error:", e);
        // Fallback to continue
        game.submitMutualStrategosChoice(1, 'continue');
        log("AI defaulted to CONTINUE.");
        render();
    } finally {
        aiStrategosSubmitted = false; // reset for next possible occurrence
    }
}

function renderGameOver() {
  reportGameResult();
  const p1Score = game.getCorePoints(0);
  const p2Score = game.getCorePoints(1);
  const p1PiecesLeft = game.players[0].units.filter(u => u.health > 0).length;
  const p2PiecesLeft = game.players[1].units.filter(u => u.health > 0).length;

  const centerControl = getCenterControl(game);
  const center = Math.floor(CONFIG.board.size / 2);
  const centerUnit = game.board.getUnitAt(center, center);
  const centerCoreCaptured = Boolean(centerUnit && centerUnit.health > 0 && centerUnit.playerId === game.winner);
  const winnerLabel = game.winner === 0 ? 'Player 1' : game.winner === 1 ? 'Player 2' : 'Draw';
  const scoreBreakdown = computeMatchScoreBreakdown({ p1Points: p1Score, p2Points: p2Score, turnCount: game.turnCount, winnerLabel, p1PiecesLeft, p2PiecesLeft, centerControl });
  const pointDifference = scoreBreakdown.rawPointDiff;
  const finalScore = scoreBreakdown.finalScore;
  saveMatchRecord({
    winner: winnerLabel,
    p1Points: p1Score,
    p2Points: p2Score,
    pointDifference,
    p1PiecesLeft,
    p2PiecesLeft,
    centerCoreOwner: centerControl.playerId,
    centerCoreWeight: centerControl.scoreWeight,
    centerCoreOccupant: centerControl.label,
    scoreBreakdown: scoreBreakdown.contributions,
    centerCoreCaptured,
    turnCount: game.turnCount,
    finalScore,
    timestamp: new Date().toLocaleString()
  });
  renderScoreboardPanel();

  let modal = document.getElementById('game-over-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'game-over-modal';
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-xl animate-in fade-in duration-500';
    document.body.appendChild(modal);
  }
  
  const isDraw = game.winner === -1;
  const isP1Win = game.winner === 0;
  const winnerText = isDraw ? "DRAW" : (isP1Win ? "VICTORY" : "DEFEAT");
  const winnerSubText = isDraw
    ? "TEMPORAL STALEMATE DECLARED"
    : (isP1Win ? "STRATEGIC DOMINANCE ACHIEVED" : "TACTICAL FAILURE DETECTED");
  const winnerColor = isDraw
    ? "text-yellow-400 drop-shadow-[0_0_20px_rgba(250,204,21,0.35)]"
    : (isP1Win ? "text-sky-blue drop-shadow-[0_0_20px_rgba(0,113,164,0.5)]" : "text-traffic-red drop-shadow-[0_0_20px_rgba(203,6,5,0.5)]");

  modal.innerHTML = `
    <div class="glass-panel p-12 rounded-[2rem] shadow-2xl flex flex-col items-center gap-8 max-w-lg w-full mx-4 relative overflow-hidden ring-1 ring-white/10">
      <div class="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent pointer-events-none"></div>
      
      <div class="flex flex-col items-center gap-2 z-10">
        <h2 class="text-7xl font-black tracking-tighter ${winnerColor}">${winnerText}</h2>
        <div class="text-[10px] font-bold text-white/50 tracking-[0.4em] uppercase mt-2">${winnerSubText}</div>
      </div>

      <div class="flex gap-12 my-2 z-10 bg-black/40 px-8 py-6 rounded-2xl border border-white/5">
         <div class="flex flex-col items-center gap-1">
            <span class="text-sky-blue font-bold text-[10px] tracking-[0.2em] uppercase">PLAYER 1</span>
            <span class="text-5xl font-mono text-white">${p1Score}</span>
         </div>
         <div class="w-px bg-white/10"></div>
         <div class="flex flex-col items-center gap-1">
            <span class="text-traffic-red font-bold text-[10px] tracking-[0.2em] uppercase">PLAYER 2</span>
            <span class="text-5xl font-mono text-white">${p2Score}</span>
         </div>
      </div>
      
      <div class="text-slate-400 text-center text-[11px] z-10 max-w-xs leading-relaxed tracking-wide">
        ${isDraw
          ? "Neither side secured decisive dominance. The timeline remains contested."
          : (isP1Win
            ? "You have successfully outmaneuvered the adversary and secured temporal control."
            : "The timeline has been compromised. The enemy has seized control of the core.")}
      </div>

      <div class="flex flex-col gap-3 w-full mt-4 z-10">
        <button id="modal-restart-btn" class="w-full py-5 bg-white text-black font-black rounded-2xl tracking-[0.3em] uppercase text-xs transition-all hover:bg-sky-blue hover:text-white active:scale-95 shadow-xl">
          Initialize New Timeline
        </button>
      </div>
    </div>
  `;
  
  document.getElementById('modal-restart-btn').onclick = () => {
    initGame();
    modal.remove();
  };
}

async function reportGameResult() {
    const winnerId = game.winner;
    const aiId = 1; // AI is Player 2
    
    let outcome = 'draw';
    if (winnerId === aiId) outcome = 'win';
    else if (winnerId !== null && winnerId !== -1) outcome = 'loss';
    
    const p1Score = game.getCorePoints(0);
    const p2Score = game.getCorePoints(1);
    const p1PiecesLeft = game.players[0].units.filter(u => u.health > 0).length;
    const p2PiecesLeft = game.players[1].units.filter(u => u.health > 0).length;
    const centerControl = getCenterControl(game);
    const winnerLabel = winnerId === 0 ? 'Player 1' : winnerId === 1 ? 'Player 2' : 'Draw';
    const finalScore = computeFinalScore({ p1Points: p1Score, p2Points: p2Score, turnCount: game.turnCount, winnerLabel, p1PiecesLeft, p2PiecesLeft, centerControl });

    const payload = {
        outcome: outcome,
        score: game.players[aiId].score,
        opponent_score: game.players[0].score,
        rounds: game.round,
        board_size: CONFIG.board.size,
        final_score: finalScore,
        reward: computeAiReward(finalScore)
    };
    
    try {
        await fetch('/api/ai/learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        console.log("Game result reported to AI.");
    } catch (e) {
        console.error("Failed to report game result:", e);
    }
}

let draggedUnit = null;

function renderBoard() {
  // Ensure layers exist
  let gridLayer = document.getElementById('grid-layer');
  let unitLayer = document.getElementById('unit-layer');
  let svgOverlay = document.getElementById('svg-overlay');

  if (!gridLayer) {
    boardEl.innerHTML = ''; // Clear init
    boardEl.style.position = 'absolute';
    boardEl.style.top = '0';
    boardEl.style.left = '0';
    boardEl.style.right = '0';
    boardEl.style.bottom = '0';
    boardEl.style.overflow = 'hidden';
    
    // Grid Layer (Background & Interaction)
    gridLayer = document.createElement('div');
    gridLayer.id = 'grid-layer';
    gridLayer.style.position = 'absolute';
    gridLayer.style.top = '0';
    gridLayer.style.left = '0';
    gridLayer.style.right = '0';
    gridLayer.style.bottom = '0';
    gridLayer.style.display = 'grid';
    gridLayer.style.gridTemplateColumns = `repeat(${CONFIG.board.size}, 1fr)`;
    gridLayer.style.gridTemplateRows = `repeat(${CONFIG.board.size}, 1fr)`;
    boardEl.appendChild(gridLayer);

    // SVG Layer
    svgOverlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgOverlay.id = 'svg-overlay';
    svgOverlay.setAttribute("class", "absolute inset-0 pointer-events-none z-20 w-full h-full");
    boardEl.appendChild(svgOverlay);

    // Unit Layer
    unitLayer = document.createElement('div');
    unitLayer.id = 'unit-layer';
    unitLayer.className = 'absolute inset-0 z-30 pointer-events-none';
    boardEl.appendChild(unitLayer);
  }

  // 1. Render Grid Cells (Interaction Targets)
  gridLayer.innerHTML = '';
  for (let r = 0; r < CONFIG.board.size; r++) {
    for (let c = 0; c < CONFIG.board.size; c++) {
      const cell = document.createElement('div');
      cell.className = 'relative flex items-center justify-center cell-transition border border-black/10';
      
      // Color Palette Logic
      const center = Math.floor(CONFIG.board.size / 2);
      if (r === center && c === center) {
        // Center Core: Brown red
        cell.style.backgroundColor = '#781c1c';
      } else if (game.board.isCoreCell(r, c)) {
        // 8 Adjacent Core: Traffic red
        cell.style.backgroundColor = '#cb0605';
      } else if (r === 0 || r === CONFIG.board.size - 1) {
        // Home Base: Black brown
        cell.style.backgroundColor = '#181818';
      } else {
        // Battlefield: Signal white
        cell.style.backgroundColor = '#efefef';
      }

      // Drag & Drop Events
      cell.ondragover = (e) => {
        e.preventDefault();
        if (draggedUnit) {
           const actions = game.getPossibleActions(draggedUnit);
           const isMove = actions.move.some(m => m.r === r && m.c === c);
           const isAttack = actions.attack.some(u => u.row === r && u.col === c);
           
           if (isMove) {
             e.dataTransfer.dropEffect = 'move';
             cell.classList.add('ring-4', 'ring-sky-blue/50', 'z-10');
           } else if (isAttack) {
             e.dataTransfer.dropEffect = 'copy';
             cell.classList.add('ring-4', 'ring-traffic-red/50', 'z-10');
           } else {
             e.dataTransfer.dropEffect = 'none';
           }
        }
      };

      cell.ondragleave = (e) => {
        cell.classList.remove('ring-4', 'ring-sky-blue/50', 'ring-traffic-red/50', 'z-10');
      };

      cell.ondrop = (e) => {
        e.preventDefault();
        cell.classList.remove('ring-4', 'ring-sky-blue/50', 'ring-traffic-red/50', 'z-10');
        if (draggedUnit) {
          handleDrop(r, c, draggedUnit);
          draggedUnit = null;
          possibleActions = null; // Clear highlights immediately
          renderBoard();
        }
      };

      // Click Event
      cell.onclick = () => handleCellClick(r, c, null);

      // Action Highlights
      if (possibleActions) {
        const moveAction = possibleActions.move.find(m => m.r === r && m.c === c);
        if (moveAction) {
          const highlight = document.createElement('div');
          highlight.className = 'absolute inset-0 bg-sky-blue/20 border-2 border-sky-blue animate-pulse-soft pointer-events-none';
          cell.appendChild(highlight);
        }

        const attackAction = possibleActions.attack.find(u => u.row === r && u.col === c);
        if (attackAction) {
          const highlight = document.createElement('div');
          highlight.className = 'absolute inset-0 bg-traffic-red/20 border-2 border-traffic-red animate-pulse-soft pointer-events-none';
          cell.appendChild(highlight);
        }
      }

      gridLayer.appendChild(cell);
    }
  }

  // 2. Render SVG Arrows
  svgOverlay.innerHTML = '';
  if (game.phase === 'planning') {
    game.timeline.forEach((slot, idx) => {
      const action = slot[0]; // P1 action
      if (action) {
        drawActionArrow(svgOverlay, action, idx);
      }
    });
  }

  // 3. Render Units (Animated)
  const activeUnitIds = new Set();
  const pct = 100 / CONFIG.board.size;

  game.players.forEach(p => {
    p.units.forEach(u => {
      if (u.health <= 0) {
        // Remove dead unit element if it exists
        const deadEl = document.getElementById(`unit-${u.id}`);
        if (deadEl) deadEl.remove();
        return;
      }
      
      activeUnitIds.add(u.id);
      let unitEl = document.getElementById(`unit-${u.id}`);
      
      if (!unitEl) {
        unitEl = createUnitElement(u);
        unitLayer.appendChild(unitEl);
      }

      // Update Position with Transition
      unitEl.style.left = `${u.col * pct}%`;
      unitEl.style.top = `${u.row * pct}%`;
      unitEl.style.width = `${pct}%`;
      unitEl.style.height = `${pct}%`;
      
      updateUnitVisuals(unitEl, u);
    });
  });

  // 4. Winning Event Highlight
  if (game.winningEvent) {
    const { type, victim, attacker, player } = game.winningEvent;
    
    if (type === 'assassination') {
      // Highlight Victim
      const vEl = document.getElementById(`unit-${victim.id}`);
      if (vEl) { // Might be removed if we cleaned up dead units, but we should keep it visible for replay?
         // Actually renderBoard cleans up dead units.
         // We need to re-create it or prevent cleanup if it's the victim in game over?
         // Or just highlight the cell.
         // Let's highlight the cell where it died.
         const cellIdx = victim.row * CONFIG.board.size + victim.col;
         const cell = gridLayer.children[cellIdx];
         if (cell) {
            const marker = document.createElement('div');
            marker.className = 'absolute inset-0 border-4 border-red-600 animate-pulse z-40 flex items-center justify-center';
            marker.innerHTML = '<span class="text-4xl">💀</span>';
            cell.appendChild(marker);
         }
      }
      
      // Highlight Attacker
      if (attacker) {
         const aEl = document.getElementById(`unit-${attacker.id}`);
         if (aEl) {
            const badge = document.createElement('div');
            badge.className = 'absolute -top-4 left-1/2 -translate-x-1/2 bg-yellow-500 text-black font-bold text-[10px] px-2 py-0.5 rounded shadow z-50 whitespace-nowrap';
            badge.textContent = 'KILLER';
            aEl.appendChild(badge);
            aEl.classList.add('ring-4', 'ring-yellow-500');
         }
      }
    } else if (type === 'core_control') {
      // Highlight Core
      for (let r = 0; r < CONFIG.board.size; r++) {
        for (let c = 0; c < CONFIG.board.size; c++) {
          if (game.board.isCoreCell(r, c)) {
             const cellIdx = r * CONFIG.board.size + c;
             const cell = gridLayer.children[cellIdx];
             if (cell) {
               cell.classList.add(player === 0 ? 'bg-sky-blue/40' : 'bg-traffic-red/40');
               cell.classList.add('animate-pulse');
             }
          }
        }
      }
    }
  }
}

function createUnitElement(unit) {
  const el = document.createElement('div');
  el.id = `unit-${unit.id}`;
  el.className = 'absolute flex items-center justify-center transition-all duration-500 ease-in-out z-10 pointer-events-auto drop-shadow-[0_4px_8px_rgba(0,0,0,0.6)]'; 
  
  const inner = document.createElement('div');
  inner.className = 'w-4/5 h-4/5 rounded-full flex items-center justify-center relative border-4 border-piece-border';
  el.appendChild(inner);

  const dot = document.createElement('div');
  dot.className = 'center-dot w-1/2 h-1/2 rounded-full bg-black flex items-center justify-center text-white font-bold text-[10px] shadow-inner';
  inner.appendChild(dot);

  // Events
  el.onclick = (e) => {
    e.stopPropagation();
    handleCellClick(unit.row, unit.col, unit);
  };

  return el;
}

function updateUnitVisuals(el, unit) {
  const inner = el.firstChild;
  const isP1 = unit.playerId === 0;
  
  // Colors & Glow
  inner.className = `w-4/5 h-4/5 rounded-full flex items-center justify-center border-4 border-piece-border relative transition-all duration-300 ${
    isP1 ? 'piece-p1' : 'piece-p2'
  }`;

  // Ensure dot exists
  let dot = inner.querySelector('.center-dot');
  if (!dot) {
    inner.textContent = ''; // Clear old text content
    dot = document.createElement('div');
    dot.className = 'center-dot w-1/2 h-1/2 rounded-full bg-black flex items-center justify-center text-white font-bold text-[10px] shadow-inner';
    inner.appendChild(dot);
  }

  // Symbol
  dot.textContent = unit.props.symbol;

  // Selection
  if (selectedUnit === unit) {
    inner.classList.add('ring-2', 'ring-white', 'scale-110', 'z-50');
  } else {
    inner.classList.remove('ring-2', 'ring-white', 'scale-110', 'z-50');
  }

  // Fatigue
  if (!unit.canAct(game.round)) {
    inner.classList.add('opacity-20', 'grayscale');
    el.draggable = false;
    el.style.cursor = 'default';
  } else {
    inner.classList.remove('opacity-20', 'grayscale');
    if (isP1 && game.phase === 'planning') {
      el.draggable = true;
      el.style.cursor = 'grab';
      
      el.ondragstart = (e) => {
        draggedUnit = unit;
        if (!selectedToken) {
           const player = game.players[0];
           const available = player.tokens.find(t => !player.usedTokens.has(t));
           if (available) {
             selectedToken = available;
             updateControls(); 
           }
        }
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', unit.id); // Required for Firefox
        
        // Visual feedback
        el.classList.add('opacity-20', 'scale-110', 'z-50');
        
        possibleActions = game.getPossibleActions(unit);
        renderBoard();
      };
      
      el.ondragend = (e) => {
        el.classList.remove('opacity-20', 'scale-110', 'z-50');
        draggedUnit = null;
        if (!selectedUnit) {
           possibleActions = null;
           renderBoard();
        }
      };
    }
  }

  // Health Bar
  // Remove old health bar if exists
  const oldHp = inner.querySelector('.hp-bar');
  if (oldHp) oldHp.remove();

  if (unit.maxHealth > 1) {
    const hpBar = document.createElement('div');
    hpBar.className = 'hp-bar absolute -bottom-1 flex gap-0.5';
    for(let i=0; i<unit.maxHealth; i++) {
      const dot = document.createElement('div');
      dot.className = `w-1.5 h-1.5 rounded-full ${i < unit.health ? (isP1 ? 'bg-sky-blue' : 'bg-traffic-red') : 'bg-white/10'}`;
      hpBar.appendChild(dot);
    }
    inner.appendChild(hpBar);
  }
}

function drawActionArrow(svg, action, slotIdx) {
  const unit = action.unit;
  const startR = unit.row;
  const startC = unit.col;
  
  // Calculate center of start cell (assuming 500px board / 7 cells)
  const cellSize = 500 / CONFIG.board.size;
  const half = cellSize / 2;
  const x1 = startC * cellSize + half;
  const y1 = startR * cellSize + half;

  let x2, y2, color;

  if (action.type === ActionType.MOVE) {
    if (action.params.target) {
      x2 = action.params.target.c * cellSize + half;
      y2 = action.params.target.r * cellSize + half;
    } else {
      // Fallback
      const { dr, dc } = action.params.direction;
      x2 = (startC + dc * unit.props.movement) * cellSize + half;
      y2 = (startR + dr * unit.props.movement) * cellSize + half;
    }
    color = '#0071a4'; // sky-blue
  } else if (action.type === ActionType.ATTACK) {
    const target = action.params.target;
    x2 = target.col * cellSize + half;
    y2 = target.row * cellSize + half;
    color = '#cb0605'; // traffic-red
  } else {
    return;
  }

  // Draw Line
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", "4");
  line.setAttribute("stroke-opacity", "0.6");
  line.setAttribute("marker-end", "url(#arrowhead)");
  
  // Add token number label
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", (x1 + x2) / 2);
  text.setAttribute("y", (y1 + y2) / 2);
  text.setAttribute("fill", "white");
  text.setAttribute("font-size", "12");
  text.setAttribute("font-weight", "bold");
  text.setAttribute("text-anchor", "middle");
  text.textContent = action.token;

  svg.appendChild(line);
  svg.appendChild(text);
  
  // Define marker if not exists
  if (!document.getElementById('arrowhead')) {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    marker.setAttribute("id", "arrowhead");
    marker.setAttribute("markerWidth", "10");
    marker.setAttribute("markerHeight", "7");
    marker.setAttribute("refX", "10");
    marker.setAttribute("refY", "3.5");
    marker.setAttribute("orient", "auto");
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", "0 0, 10 3.5, 0 7");
    polygon.setAttribute("fill", "#fff");
    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);
  }
}

function handleDrop(r, c, unit) {
  if (!selectedToken) {
    log("No token selected!");
    return;
  }
  
  const actions = game.getPossibleActions(unit);
  
  // Check Move
  const moveAction = actions.move.find(m => m.r === r && m.c === c);
  if (moveAction) {
      const success = game.placeAction(0, game.currentSlot, selectedToken, unit, ActionType.MOVE, { target: {r, c} });
      if (success) {
        log(`Placed MOVE with token ${selectedToken}`);
        resetSelection();
        render();
      }
      return;
  }

  // Check Attack
  const attackAction = actions.attack.find(u => u.row === r && u.col === c);
  if (attackAction) {
      const success = game.placeAction(0, game.currentSlot, selectedToken, unit, ActionType.ATTACK, { target: attackAction });
      if (success) {
        log(`Placed ATTACK with token ${selectedToken}`);
        resetSelection();
        render();
      }
      return;
  }
}

function renderTimeline() {
  timelineEl.innerHTML = '';
  game.timeline.forEach((slot, idx) => {
    const slotEl = document.createElement('div');
    slotEl.className = `flex flex-col gap-1 p-2 border ${idx === game.currentSlot ? 'border-sky-blue bg-sky-blue/10' : 'border-slate-700 bg-slate-800'} rounded min-h-[80px] min-w-[80px] items-center justify-center relative transition-colors`;
    
    // Slot Number
    const num = document.createElement('span');
    num.textContent = idx + 1;
    num.className = 'absolute top-1 left-2 text-[10px] text-slate-500 font-mono';
    slotEl.appendChild(num);

    // P1 Action
    if (slot[0]) {
      const p1Token = document.createElement('div');
      p1Token.className = 'w-6 h-6 rounded bg-sky-blue/20 border border-sky-blue flex items-center justify-center text-xs text-sky-blue font-bold';
      // Always show P1 (Human) tokens, or if resolved
      p1Token.textContent = slot[0].token;
      slotEl.appendChild(p1Token);
    }

    // P2 Action
    if (slot[1]) {
      const p2Token = document.createElement('div');
      p2Token.className = 'w-6 h-6 rounded bg-traffic-red/20 border border-traffic-red flex items-center justify-center text-xs text-traffic-red font-bold';
      // Only show P2 (AI) tokens if resolved or game over
      const isRevealed = game.phase === 'game_over' || (game.phase === 'resolution' && idx < game.currentSlot); 
      p2Token.textContent = isRevealed ? slot[1].token : '?';
      slotEl.appendChild(p2Token);
    }

    timelineEl.appendChild(slotEl);
  });
}

function renderInfo() {
  phaseIndicatorEl.textContent = game.phase.toUpperCase();
  turnIndicatorEl.textContent = game.phase === 'planning'
    ? `PLAYER ${game.currentPlayerId + 1} PLANNING (SLOT ${Math.min(game.currentSlot + 1, game.timeline.length)} / ${game.timeline.length})`
    : game.phase === 'strategos_decision'
      ? 'MUTUAL STRATEGOS DECISION PENDING'
      : `RESOLVING SLOT ${Math.min(game.currentSlot + 1, game.timeline.length)} / ${game.timeline.length}`;
  
  turnIndicatorEl.className = game.currentPlayerId === 0 ? 'text-sky-blue' : 'text-traffic-red';

  // Update Score Display (Core Points are what matter for win condition)
  const p1CorePoints = game.getCorePoints(0);
  const p2CorePoints = game.getCorePoints(1);
  
  const scoreDisplay = document.getElementById('score-display');
  if (scoreDisplay) {
    scoreDisplay.innerHTML = `
      <div class="flex flex-col items-center">
        <span class="text-[10px] text-slate-500 font-bold tracking-widest">CORE POINTS</span>
        <div class="flex gap-4 text-xl font-mono font-bold">
          <div class="text-sky-blue">P1: ${p1CorePoints}</div>
          <div class="text-slate-600">/</div>
          <div class="text-traffic-red">P2: ${p2CorePoints}</div>
        </div>
        <div class="text-[10px] text-slate-600 mt-1">TARGET: 5+</div>
        <div class="text-[10px] text-slate-500 mt-1">ROUNDS COMPLETED: <span class="font-mono text-white">${game.turnCount}</span></div>
      </div>
    `;
  }
}

function updateControls() {
  tokenContainerEl.innerHTML = '';
  actionButtonsEl.innerHTML = '';

  if (game.phase === 'game_over') {
    const btn = document.createElement('button');
    btn.textContent = 'NEW GAME';
    btn.className = 'px-4 py-2 bg-white text-black hover:bg-sky-blue hover:text-white rounded font-bold tracking-wider transition-colors';
    btn.onclick = initGame;
    actionButtonsEl.appendChild(btn);
    return;
  }

  if (game.phase === 'strategos_decision') {
    const prompt = document.createElement('div');
    prompt.className = 'text-[11px] text-slate-400 leading-relaxed';
    prompt.textContent = 'Both Strategos units were eliminated. Select your secret choice.';
    actionButtonsEl.appendChild(prompt);

    const continueBtn = document.createElement('button');
    continueBtn.textContent = 'CONTINUE';
    continueBtn.className = 'px-4 py-2 rounded font-bold tracking-wider w-full bg-sky-blue/20 border border-sky-blue text-sky-blue hover:bg-sky-blue hover:text-white';
    continueBtn.onclick = () => {
      const ok = game.submitMutualStrategosChoice(0, 'continue');
      if (ok) {
        log('Secret choice submitted: CONTINUE.');
        render();
      }
    };
    actionButtonsEl.appendChild(continueBtn);

    const endBtn = document.createElement('button');
    endBtn.textContent = 'END';
    endBtn.className = 'px-4 py-2 rounded font-bold tracking-wider w-full bg-yellow-600/20 border border-yellow-500 text-yellow-300 hover:bg-yellow-500 hover:text-black';
    endBtn.onclick = () => {
      const ok = game.submitMutualStrategosChoice(0, 'end');
      if (ok) {
        log('Secret choice submitted: END.');
        render();
      }
    };
    actionButtonsEl.appendChild(endBtn);
    return;
  }

  if (game.phase === 'resolution') {
    const btn = document.createElement('button');
    btn.textContent = 'RESOLVE NEXT';
    btn.className = 'px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded font-bold tracking-wider w-full';
    btn.onclick = () => {
      const res = game.resolveNextSlot();
      if (res) {
        res.results.forEach(r => log(r));
        render();
      }
    };
    actionButtonsEl.appendChild(btn);
    return;
  }

  // Planning Phase
  const player = game.players[game.currentPlayerId];
  
  // AI Turn Check
  if (game.currentPlayerId === 1) { // AI is Player 2
    const btn = document.createElement('button');
    btn.textContent = 'AI MOVE';
    btn.className = 'px-4 py-2 bg-traffic-red/20 border border-traffic-red text-traffic-red rounded font-bold tracking-wider w-full animate-pulse-soft';
    btn.onclick = async () => {
      btn.disabled = true;
      btn.textContent = 'THINKING...';
      
      try {
        const state = game.serialize();
        // Use full URL if needed, but relative should work with proxy
        const response = await fetch('/api/ai/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        if (data.move) {
            const success = game.executeRemoteMove(data.move);
            if (success) {
                log("AI placed an action.");
                selectedUnit = null;
                selectedToken = null;
                possibleActions = null;
                render();
            } else {
                log("AI move failed execution. Fallback to local.");
                game.aiMove(1);
                render();
            }
        } else {
            log("AI passed. Fallback to local.");
            game.aiMove(1);
            render();
        }
      } catch (e) {
        console.error("AI Error:", e);
        log(`AI Error: ${e.message}. Fallback to local.`);
        const success = game.aiMove(1);
        if (success) {
            log("AI placed an action (Local).");
            render();
        }
      } finally {
        btn.disabled = false;
        btn.textContent = 'AI MOVE';
      }
    };
    actionButtonsEl.appendChild(btn);
    return;
  }

  // Human Turn
  const humanPlayer = game.players[0];
  const actedUnits = new Set();
  game.timeline.forEach(([a0, a1]) => {
    if (a0 && a0.unit && a0.unit.playerId === 0) actedUnits.add(a0.unit);
    if (a1 && a1.unit && a1.unit.playerId === 0) actedUnits.add(a1.unit);
  });

  const activeHumanUnits = humanPlayer.getActiveUnits(game.round).filter(u => !actedUnits.has(u));

  const reducedActionMode = game.requiredActionsPerPlayer[0] < 5;

  if (selectedToken && !reducedActionMode) {
    const passBtn = document.createElement('button');
    passBtn.textContent = 'PASS TURN';
    passBtn.className = 'px-4 py-2 rounded font-bold tracking-wider w-full bg-slate-700 hover:bg-slate-600 text-white';
    passBtn.onclick = () => {
      const success = game.placeAction(0, game.currentSlot, selectedToken, null, ActionType.PASS, {});
      if (success) {
        log(`Placed PASS with token ${selectedToken}`);
        resetSelection();
        render();
      }
    };
    actionButtonsEl.appendChild(passBtn);
  }

  if (!selectedUnit) {
    const hint = document.createElement('div');
    hint.textContent = activeHumanUnits.length
      ? "Select a unit to act."
      : (reducedActionMode
        ? "Fewer than 5 active pieces: each remaining piece must act this round."
        : "No units can act this slot. Select a token and use PASS TURN.");
    hint.className = "text-slate-400 text-sm text-center italic";
    actionButtonsEl.appendChild(hint);
    return;
  }

  // Token Selection
  player.tokens.forEach(t => {
    const btn = document.createElement('button');
    btn.textContent = t;
    const isUsed = player.usedTokens.has(t);
    const isSelected = selectedToken === t;
    
    btn.className = `w-10 h-10 rounded-xl font-bold flex items-center justify-center transition-all ${
      isUsed 
        ? 'bg-white/5 text-slate-600 cursor-not-allowed' 
        : isSelected
          ? 'bg-sky-blue text-white ring-2 ring-white scale-110 shadow-lg shadow-sky-blue/50'
          : 'bg-white/10 text-sky-blue hover:bg-white/20'
    }`;
    
    if (!isUsed) {
      btn.onclick = () => {
        selectedToken = t;
        updateControls();
      };
    }
    tokenContainerEl.appendChild(btn);
  });

  // Claim Button (if applicable)
  if (possibleActions && possibleActions.claim) {
    const btn = document.createElement('button');
    btn.textContent = 'CLAIM CORE';
    btn.className = `px-4 py-2 rounded font-bold tracking-wider w-full ${
      selectedToken 
        ? 'bg-yellow-600 hover:bg-yellow-500 text-white' 
        : 'bg-slate-700 text-slate-500 cursor-not-allowed'
    }`;
    btn.onclick = () => {
      if (selectedToken) {
        const success = game.placeAction(0, game.currentSlot, selectedToken, selectedUnit, ActionType.CLAIM, {});
        if (success) {
          log(`Placed CLAIM with token ${selectedToken}`);
          resetSelection();
          render();
        }
      }
    };
    actionButtonsEl.appendChild(btn);
  }
}

function handleCellClick(r, c, unit) {
  if (game.phase !== 'planning' || game.currentPlayerId !== 0) return;

  // If clicking on an action highlight (Move/Attack)
  if (selectedUnit && selectedToken && possibleActions) {
    const moveAction = possibleActions.move.find(m => m.r === r && m.c === c);
    if (moveAction) {
      // Execute Move
      const success = game.placeAction(0, game.currentSlot, selectedToken, selectedUnit, ActionType.MOVE, { target: {r, c} });
      if (success) {
        log(`Placed MOVE with token ${selectedToken}`);
        resetSelection();
        render();
        return;
      }
    }

    const attackAction = possibleActions.attack.find(u => u.row === r && u.col === c);
    if (attackAction) {
      // Execute Attack
      const success = game.placeAction(0, game.currentSlot, selectedToken, selectedUnit, ActionType.ATTACK, { target: attackAction });
      if (success) {
        log(`Placed ATTACK with token ${selectedToken}`);
        resetSelection();
        render();
        return;
      }
    }
  }

  // Select Unit
  if (unit && unit.playerId === 0 && unit.canAct(game.round)) {
    selectedUnit = unit;
    possibleActions = game.getPossibleActions(unit);
    // Keep token if selected, else null
    render();
  } else if (unit === null) {
    // Deselect if clicking empty space (and not moving)
    if (!possibleActions || !possibleActions.move.find(m => m.r === r && m.c === c)) {
       resetSelection();
       render();
    }
  }
}

function resetSelection() {
  selectedUnit = null;
  selectedToken = null;
  possibleActions = null;
}

// Start
if (boardSizeSelectEl) {
  boardSizeSelectEl.innerHTML = BOARD_VARIANTS.map(size => `<option value="${size}">${size} x ${size}</option>`).join('');
  boardSizeSelectEl.value = String(CONFIG.board.size);
}

initGame(CONFIG.board.size);
renderScoreboardPanel();
switchSidePanel(activeSidePanel);
setSidebarVisibility(false);

if (commsTabBtn && scoreboardTabBtn) {
  commsTabBtn.onclick = () => switchSidePanel('comms');
  scoreboardTabBtn.onclick = () => switchSidePanel('scoreboard');
}

if (boardSizeSelectEl) {
  boardSizeSelectEl.onchange = (event) => {
    const selectedSize = Number(event.target.value);
    if (!BOARD_VARIANTS.includes(selectedSize)) return;
    initGame(selectedSize);
  };
}

if (sidebarToggleBtn) {
  sidebarToggleBtn.onclick = () => setSidebarVisibility(!sidePanelHidden);
}

// Chat Logic
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send');
const chatHistory = document.getElementById('chat-history');

let chatSession = null;

if (chatInput && chatSendBtn && chatHistory) {
  chatSendBtn.onclick = sendChatMessage;
  chatInput.onkeypress = (e) => {
    if (e.key === 'Enter') sendChatMessage();
  };
}

async function sendChatMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  // User Message
  appendChatMessage('You', text, 'text-sky-blue');
  chatInput.value = '';

  // Loading state
  const loadingId = appendChatMessage('AI', 'Thinking...', 'text-traffic-red animate-pulse-soft');

  try {
    if (!chatSession) {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) throw new Error("Missing API Key");
        
        const ai = new GoogleGenAI({ apiKey });
        chatSession = ai.chats.create({
            model: "gemini-3-flash-preview",
            config: {
                systemInstruction: "You are a tactical AI assistant for the game Chronos: Clash of Wills. Help the player with strategy. Keep responses concise and tactical.",
            }
        });
    }

    const result = await chatSession.sendMessage({ message: text });
    const responseText = result.text;

    // Remove loading message
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) loadingEl.remove();

    appendChatMessage('AI', responseText, 'text-traffic-red');

  } catch (e) {
    console.error(e);
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) loadingEl.remove();
    appendChatMessage('System', `Connection Lost. ${e.message}`, 'text-red-500');
  }
}

function appendChatMessage(sender, text, colorClass) {
  const id = 'msg-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'flex flex-col gap-1';
  div.innerHTML = `
    <span class="font-bold text-[10px] uppercase tracking-wider ${colorClass}">${sender}</span>
    <div class="text-slate-300 leading-relaxed">${text}</div>
  `;
  chatHistory.appendChild(div);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return id;
}
