import { getAllPossibleMoves, getBestMove } from '../utils/aiLogic.js';

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
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const response = await fetch('/api/ai/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...gameState, validMoves }),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) throw new Error(`AI endpoint failed (${response.status})`);

    const payload = await response.json();
    const move = payload.move || payload.choice || null;

    if (!isValidMove(move)) return localFallback();
    return move;
  } catch (error) {
    console.warn('Falling back to local Aether AI:', error);
    return localFallback();
  }
};