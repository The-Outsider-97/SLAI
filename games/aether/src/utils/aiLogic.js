import { BOARD_SIZE, POWER_WELLS } from '../constants.js';
import { isValidMove, executeAction, getPathProgress } from './gameLogic.js';

export const getAllPossibleMoves = (gameState) => {
  const moves = [];
  const { faceUpCards = [] } = gameState || {};

  for (const card of faceUpCards) {
    for (let i = 0; i < (card.actions || []).length; i++) {
      const action = card.actions[i];
      for (const target of getValidTargets(gameState, action)) {
        moves.push({ cardId: card.id, actionIndex: i, target });
      }
    }
  }

  return moves;
};

const getValidTargets = (gameState, action) => {
  const targets = [];
  const { board, activePlayer, players } = gameState;
  const player = players?.[activePlayer];
  if (!player) return targets;

  const add = (r, c) => targets.push({ row: r, col: c });

  switch (action) {
    case 'PLACE':
      if (player.position) {
        const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
        for (const [dr, dc] of dirs) {
          const r = player.position.row + dr;
          const c = player.position.col + dc;
          if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && !board[r][c]) add(r, c);
        }
      }
      break;
    case 'ROTATE':
      for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
          const tile = board[r][c];
          if (tile && (!tile.hasResonator || tile.resonatorOwner === activePlayer)) add(r, c);
        }
      }
      break;
    case 'ADVANCE':
      if (player.position) {
        for (let r = 0; r < BOARD_SIZE; r++) for (let c = 0; c < BOARD_SIZE; c++) {
          if (isValidMove(board, player.position, { row: r, col: c })) add(r, c);
        }
      }
      break;
    case 'ATTUNE':
      if (player.position) {
        const { row, col } = player.position;
        const tile = board[row][col];
        if (tile && !tile.hasResonator && player.resonators > 0) add(row, col);
      }
      break;
    case 'SHIFT':
      for (let c = 0; c < BOARD_SIZE; c++) add(0, c);
      for (let c = 0; c < BOARD_SIZE; c++) add(BOARD_SIZE - 1, c);
      for (let r = 1; r < BOARD_SIZE - 1; r++) add(r, 0);
      for (let r = 1; r < BOARD_SIZE - 1; r++) add(r, BOARD_SIZE - 1);
      break;
  }

  return targets;
};

export const getBestMove = (gameState) => {
  const moves = getAllPossibleMoves(gameState);
  if (!moves.length) return null;

  let bestMove = null;
  let bestScore = -Infinity;

  for (const move of moves) {
    const card = gameState.faceUpCards.find((c) => c.id === move.cardId);
    if (!card) continue;
    const action = card.actions[move.actionIndex];
    const result = executeAction(gameState, action, move.target);
    if (!result.success) continue;

    const score = evaluateState(result.newState, gameState.activePlayer);
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }

  return bestMove || moves[Math.floor(Math.random() * moves.length)];
};

const evaluateState = (gameState, playerId) => {
  if (gameState.winner === playerId) return 1000000;
  if (gameState.winner && gameState.winner !== playerId) return -1000000;

  const player = gameState.players[playerId];
  const opponentId = playerId === 1 ? 2 : 1;

  const myWells = Object.values(gameState.capturedWells || {}).filter((id) => id === playerId).length;
  const oppWells = Object.values(gameState.capturedWells || {}).filter((id) => id === opponentId).length;

  let score = myWells * 5000 - oppWells * 6000;
  score += getPathProgress(gameState, playerId) * 100;
  score -= getPathProgress(gameState, opponentId) * 120;

  if (player?.position) {
    const dist = Math.abs(player.position.row - player.goalRow);
    score += (BOARD_SIZE - dist) * 200;
    const wellKey = `${player.position.row},${player.position.col}`;
    const onWell = POWER_WELLS.some((w) => w.row === player.position.row && w.col === player.position.col);
    if (onWell && !gameState.capturedWells[wellKey]) score += 1000;
  }

  return score;
};