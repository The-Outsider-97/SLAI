import { getAllPossibleMoves, getBestMove } from '../utils/aiLogic.js';

const AETHER_GAME_KEY = 'aether_shift';
const AI_MOVE_TIMEOUT_MS = 15000;
const DEBUG_STORAGE_KEY = 'aether_shift_last_ai_debug_v2';

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

const sameMove = (left, right) => (
  !!left
  && !!right
  && left.cardId === right.cardId
  && left.actionIndex === right.actionIndex
  && left.target?.row === right.target?.row
  && left.target?.col === right.target?.col
);

const storeDebugPayload = (payload) => {
  if (!payload || typeof payload !== 'object') return;
  try {
    const debugPayload = {
      receivedAt: new Date().toISOString(),
      confidence: payload.confidence ?? null,
      strategy: payload.strategy ?? null,
      reasoning: payload.reasoning ?? null,
      fallback: Boolean(payload.fallback),
      fallback_reason: payload.fallback_reason ?? null,
      agent_trace: payload.agent_trace ?? null,
      debug: payload.debug ?? null,
    };
    window.__AETHER_LAST_AI_DEBUG__ = debugPayload;
    localStorage.setItem(DEBUG_STORAGE_KEY, JSON.stringify(debugPayload));
  } catch {
    // Debug storage must never break gameplay.
  }
};

export const requestAetherMove = async (gameState) => {
  const validMoves = getAllPossibleMoves(gameState);
  if (!validMoves.length) return null;

  const isValidMove = (candidate) => validMoves.some((move) => sameMove(move, candidate));

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
    storeDebugPayload(payload);

    // Compatibility order:
    // 1. shared backend: { move: rawMove }
    // 2. thin bridge / future endpoint: { move: rawMove, confidence, agent_trace }
    // 3. debug wrapper: { move_response: { move: rawMove } }
    // 4. legacy choice fallback.
    const candidate = payload?.move_response?.move || payload?.move || payload?.choice || null;

    if (!isValidMove(candidate)) return localFallback();
    return candidate;
  } catch (error) {
    if (error?.name === 'AbortError') {
      console.warn(`Aether AI request timed out after ${AI_MOVE_TIMEOUT_MS}ms, using fallback.`);
      return localFallback();
    }
    console.warn('Falling back to local Aether AI:', error);
    return localFallback();
  }
};

export const getLastAetherAiDebug = () => {
  if (window.__AETHER_LAST_AI_DEBUG__) return window.__AETHER_LAST_AI_DEBUG__;
  try {
    const raw = localStorage.getItem(DEBUG_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};
