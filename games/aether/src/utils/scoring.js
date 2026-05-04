export const SCOREBOARD_STORAGE_KEY = 'aether_shift_scoreboard_v1';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const getWellControlStats = (gameState, winnerId) => {
  const winnerColor = gameState.players[winnerId]?.color;
  let ownTileWells = 0;
  let enemyTileWells = 0;

  for (const well of gameState.powerWells) {
    const key = `${well.row},${well.col}`;
    if (gameState.capturedWells[key] !== winnerId) continue;

    const tileColor = gameState.board[well.row]?.[well.col]?.color;
    if (tileColor === winnerColor) ownTileWells += 1;
    else if (tileColor && tileColor !== 'neutral') enemyTileWells += 1;
  }

  return { ownTileWells, enemyTileWells };
};

const getScoreFromWinMethod = (gameState, winnerId) => {
  if (gameState.winReason === 'Path Completed!') {
    return { score: 6, reason: 'Edge connection' };
  }

  const { ownTileWells, enemyTileWells } = getWellControlStats(gameState, winnerId);
  const totalWinnerWells = ownTileWells + enemyTileWells;

  if (totalWinnerWells === 0) {
    return { score: 3, reason: 'Well capture' };
  }

  const ownRatio = ownTileWells / totalWinnerWells;
  const score = Math.round(3 + ownRatio * 2); // 3..5 based on own tile usage
  return { score, reason: 'Well capture' };
};

const getPoints = (gameState, winnerId, score) => {
  const turnsUsed = Number.isFinite(gameState.turn) ? gameState.turn : 1;
  const speedBonus = clamp((24 - turnsUsed) * 3, 0, 60);
  const methodBase = gameState.winReason === 'Path Completed!' ? 28 : 16;

  const { ownTileWells, enemyTileWells } = getWellControlStats(gameState, winnerId);
  const tileControlBonus = clamp((ownTileWells * 7) - (enemyTileWells * 3), -10, 20);

  const normalizedScoreBonus = (score - 3) * 8;
  const winnerPoints = clamp(Math.round(methodBase + speedBonus + tileControlBonus + normalizedScoreBonus), 1, 100);

  return winnerPoints;
};

export const computeMatchRecord = (gameState) => {
  const redWells = Object.values(gameState.capturedWells).filter((id) => id === 1).length;
  const blueWells = Object.values(gameState.capturedWells).filter((id) => id === 2).length;
  const turns = Number.isFinite(gameState.turn) ? gameState.turn : 1;

  const winnerId = gameState.winner;
  const winner = winnerId ? gameState.players[winnerId].name : 'In Progress';

  const { score } = winnerId ? getScoreFromWinMethod(gameState, winnerId) : { score: 0 };
  const winnerPoints = winnerId ? getPoints(gameState, winnerId, score) : 0;

  return {
    id: `match-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    timestamp: new Date().toLocaleString(),
    mode: gameState.mode === 'PVAI' ? 'Player v. AI' : 'Player v. Player',
    turns,
    rounds: Math.ceil(turns / 2),
    wells: { red: redWells, blue: blueWells },
    winner,
    score,
    points: winnerId === 1 ? winnerPoints : winnerId === 2 ? -winnerPoints : 0,
  };
};
