
const PIECE_VALUES = {
  'Strategos': 3,
  'Warden': 2,
  'Scout': 1
};

class AIPlayer {
  constructor() {
    console.log("AI Player initialized (Node.js version)");
  }

  getMove(gameState) {
    try {
      const validMoves = gameState.validMoves || [];
      if (!validMoves.length) {
        console.warn("No valid moves provided in game state.");
        return null;
      }

      // Simple heuristic scoring
      let bestMove = null;
      let maxScore = -Infinity;

      for (const move of validMoves) {
        const score = this._scoreMove(move, gameState);
        if (score > maxScore) {
          maxScore = score;
          bestMove = move;
        }
      }

      console.log(`AI selected move with score ${maxScore}:`, bestMove);
      return bestMove;

    } catch (e) {
      console.error("Error in getMove:", e);
      // Fallback to random
      return gameState.validMoves && gameState.validMoves.length > 0 
        ? gameState.validMoves[Math.floor(Math.random() * gameState.validMoves.length)] 
        : null;
    }
  }

  _scoreMove(move, gameState) {
    let score = 0;
    
    const moveType = move.type;
    const params = move.params || {};
    const target = move.target || params.target || {}; // Handle both structures
    const tr = target.r;
    const tc = target.c;
    const unitId = move.unitId;

    // Find acting unit
    let actingUnit = null;
    // gameState.units is array of {id, type, ...}
    if (gameState.units) {
        actingUnit = gameState.units.find(u => u.id === unitId);
    }
    
    if (!actingUnit) return -1000;

    // 1. Core Control (High Priority)
    if (tr === 4 && tc === 4) {
      score += 100; // Center core
    } else if (this._isCoreCell(tr, tc)) {
      score += 40; // Adjacent core
    }

    // 2. Attack (High Priority)
    if (moveType === 'attack') {
      // Target might be a unit object or just coords?
      // In engine.js getValidMovesForAI:
      // moves.push({ ..., target: targetUnit }) where targetUnit is the unit object
      // So target.type should exist
      const targetType = target.type;
      const targetValue = PIECE_VALUES[targetType] || 1;
      score += 50 * targetValue;

      if (targetType === 'Strategos') {
        score += 1000; // Winning move
      }
    }

    // 3. Piece Protection & Value
    const unitType = actingUnit.type;
    
    if (unitType === 'Strategos') {
      // Keep Strategos safe but active
      if (this._isCoreCell(tr, tc)) {
        score += 30;
      } else {
        score += 10;
      }
    }

    // 4. Random factor
    score += Math.random() * 5;

    return score;
  }

  _isCoreCell(r, c) {
    // 3x3 core in center (3,3 to 5,5)
    return r >= 3 && r <= 5 && c >= 3 && c <= 5;
  }
}

export const aiPlayer = new AIPlayer();
