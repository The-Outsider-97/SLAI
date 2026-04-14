import { getAllPossibleMoves, getBestMove } from '../utils/aiLogic.js';

const AETHER_GAME_KEY = 'aether_shift';
const AI_MOVE_TIMEOUT_MS = 15000;

let runtimeSelectionPromise = null;

const ensureAetherRuntimeSelected = async () => {
  if (runtimeSelectionPromise) return runtimeSelectionPromise;

  runtimeSelectionPromise = fetch('/api/select-game', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ game: AETHER_GAME_KEY }),
  })
    .then(async (response) => {
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `select-game failed (${response.status})`);
      }
      return response.json();
    })
    .catch((error) => {
      runtimeSelectionPromise = null;
      throw error;
    });

  return runtimeSelectionPromise;
};

export const requestAetherMove = async (gameState) => {
  const validMoves = getAllPossibleMoves(gameState);
  if (!validMoves.length) return null;

  const isValidMove = (candidate) => {
    if (!candidate) return false;
    return validMoves.some(
      (move) =>
        move.cardId === candidate.cardId
        && move.actionIndex === candidate.actionIndex
        && move.target?.row === candidate.target?.row
        && move.target?.col === candidate.target?.col,
    );
  };

  const localFallback = () => {
    const bestMove = getBestMove(gameState);
    if (isValidMove(bestMove)) return bestMove;
    return validMoves[0] || null;
  };

  try {
    await ensureAetherRuntimeSelected();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), AI_MOVE_TIMEOUT_MS);
    let response;
    try {
      response = await fetch('/api/ai/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...gameState, game: AETHER_GAME_KEY, validMoves }),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeout);
    }

    if (!response.ok) throw new Error(`AI endpoint failed (${response.status})`);

    const payload = await response.json();
    const move = payload.move || payload.choice || null;

    if (!isValidMove(move)) return localFallback();
    return move;
  } catch (error) {
    if (error?.name === 'AbortError') {
      console.warn(`Aether AI request timed out after ${AI_MOVE_TIMEOUT_MS}ms, using fallback.`);
      return localFallback();
    }
    console.warn('Falling back to local Aether AI:', error);
    return localFallback();
  }
};
