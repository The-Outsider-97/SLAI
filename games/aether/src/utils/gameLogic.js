import { BOARD_SIZE, POWER_WELLS, TILE_CONNECTIONS, generateDeck } from '../constants.js';

const isPowerWell = (row, col) => POWER_WELLS.some((well) => well.row === row && well.col === col);

const refreshCapturedWells = (state) => {
  const captured = {};
  for (const well of POWER_WELLS) {
    const tile = state.board?.[well.row]?.[well.col];
    if (tile?.hasResonator && tile.resonatorOwner) {
      captured[`${well.row},${well.col}`] = tile.resonatorOwner;
    }
  }
  state.capturedWells = captured;
};

export const createInitialState = ({ mode = 'PVAI', aiStarts = false } = {}) => {
  const board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));

  // Initialize players
  const players = {
    1: {
      id: 1,
      name: 'Red Architect',
      color: 'red',
      homeRow: 0,
      goalRow: 4,
      resonators: 4,
      position: { row: 0, col: 2 }, // Start middle of home row
    },
    2: {
      id: 2,
      name: 'Blue Architect',
      color: 'blue',
      homeRow: 4,
      goalRow: 0,
      resonators: 4,
      position: { row: 4, col: 2 }, // Start middle of home row
    },
  };

  // Place initial starting tiles for players
  board[0][2] = createTile('STRAIGHT', 0, 'red');
  board[0][2].playersPresent = [1];
  
  board[4][2] = createTile('STRAIGHT', 0, 'blue');
  board[4][2].playersPresent = [2];

  // Add some random neutral tiles to start with so the board isn't empty
  board[2][2] = createTile('CROSS', 0, 'neutral');
  board[1][1] = createTile('CURVE', 90, 'neutral');
  board[1][3] = createTile('CURVE', 180, 'neutral');
  board[3][1] = createTile('CURVE', 0, 'neutral');
  board[3][3] = createTile('CURVE', 270, 'neutral');


  const deck = generateDeck();
  // Use the full fixed deck as faceUpCards
  const faceUpCards = [...deck];

  return {
    mode,
    board,
    players,
    activePlayer: aiStarts ? 2 : 1,
    turn: 1,
    actionsRemaining: 2,
    deck: [], // Deck is empty as all cards are face up
    faceUpCards,
    selectedCardId: null,
    selectedActionIndex: null,
    winner: null,
    winReason: null,
    powerWells: POWER_WELLS,
    capturedWells: {},
    showGuide: true,
  };
};

export const createTile = (type, rotation, color) => {
  return {
    id: `tile-${Date.now()}-${Math.random()}`,
    type,
    rotation,
    color,
    hasResonator: false,
    playersPresent: [],
  };
};

// --- Logic Helpers ---


const isAdjacentToPlayer = (playerPosition, target) => {
  if (!playerPosition || !target) return false;
  const rowDelta = Math.abs(playerPosition.row - target.row);
  const colDelta = Math.abs(playerPosition.col - target.col);
  return rowDelta + colDelta === 1;
};

export const executeAction = (gameState, action, target) => {
    // Deep clone state to prevent mutation
    let newState = JSON.parse(JSON.stringify(gameState));
    let success = false;
    let message = '';

    const { board, activePlayer, players } = newState;
    const player = players[activePlayer];

    if (!player) {
        return { newState: gameState, success: false, message: 'Invalid player.' };
    }

    // Validate target bounds
    if (!target || typeof target.row !== 'number' || typeof target.col !== 'number' ||
        target.row < 0 || target.row >= BOARD_SIZE || 
        target.col < 0 || target.col >= BOARD_SIZE) {
        return { newState: gameState, success: false, message: 'Target out of bounds or invalid.' };
    }

    switch (action) {
      case 'PLACE':
        if (board[target.row][target.col]) {
          message = 'Slot is not empty.';
          break;
        }

        if (!isAdjacentToPlayer(player.position, target)) {
          message = 'Place must be on an empty tile adjacent to your Architect.';
          break;
        }

        const types = ['STRAIGHT', 'CURVE', 'T_JUNCTION'];
        const randomType = types[Math.floor(Math.random() * types.length)];
        board[target.row][target.col] = createTile(randomType, 0, player.color);
        success = true;
        message = 'Tile placed.';
        break;

      case 'ROTATE':
        const tileToRotate = board[target.row][target.col];
        if (tileToRotate) {
          if (tileToRotate.hasResonator && tileToRotate.resonatorOwner !== activePlayer) {
             message = 'Cannot rotate a tile locked by opponent.';
          } else {
            tileToRotate.rotation = (tileToRotate.rotation + 90) % 360 ;
            success = true;
            message = 'Tile rotated.';
          }
        } else {
          message = 'No tile to rotate.';
        }
        break;

      case 'ADVANCE':
        // Move meeple
        if (player.position) {
            // Check if target is adjacent and connected
            if (isValidMove(board, player.position, target)) {
                // Remove from old tile
                const oldTile = board[player.position.row][player.position.col];
                if (oldTile) {
                    oldTile.playersPresent = oldTile.playersPresent.filter(id => id !== activePlayer);
                }
                
                // Update player position
                player.position = target;
                
                // Add to new tile
                const newTile = board[target.row][target.col];
                if (newTile) {
                    newTile.playersPresent.push(activePlayer);
                }
                success = true;
                message = 'Architect moved.';
            } else {
                message = 'Invalid move. Path not connected.';
            }
        } else {
            // Player not on board? (Shouldn't happen with init state)
        }
        break;

      case 'ATTUNE':
        const tileToAttune = board[target.row][target.col];
        if (tileToAttune) {
            if (player.position?.row === target.row && player.position?.col === target.col) {
                if (!tileToAttune.hasResonator) {
                    if (player.resonators > 0) {
                        const isWell = isPowerWell(target.row, target.col);
                        if (isWell && tileToAttune.color === 'neutral') {
                            message = 'Cannot attune an existing grey tile to capture a Power Well.';
                            break;
                        }

                        tileToAttune.hasResonator = true;
                        tileToAttune.resonatorOwner = activePlayer;
                        player.resonators--;

                        if (isWell) {
                            message = 'Resonator placed. Power Well captured!';
                        } else {
                            message = 'Resonator placed.';
                        }
                        success = true;
                    } else {
                        message = 'No resonators remaining.';
                    }
                } else {
                    message = 'Tile already has a resonator.';
                }
            } else {
                message = 'You must be on the tile to attune it.';
            }
        }
        break;
        
      case 'SHIFT':
        // Determine axis and direction
        let axis = null;
        let direction = null; // 1 = right/down, -1 = left/up
        
        if (target.row === 0) { axis = 'col'; direction = 1; } // Top edge -> Shift Down
        else if (target.row === BOARD_SIZE - 1) { axis = 'col'; direction = -1; } // Bottom edge -> Shift Up
        else if (target.col === 0) { axis = 'row'; direction = 1; } // Left edge -> Shift Right
        else if (target.col === BOARD_SIZE - 1) { axis = 'row'; direction = -1; } // Right edge -> Shift Left
        
        if (axis && direction) {
            // Perform shift
            if (axis === 'row') {
                const row = board[target.row];
                if (direction === 1) { // Right
                    const popped = row.pop();
                    row.unshift(popped);
                } else { // Left
                    const popped = row.shift();
                    row.push(popped);
                }
                // Update player positions if they were on this row
                Object.values(players).forEach(p => {
                    if (p.position && p.position.row === target.row) {
                        let newCol = p.position.col + direction;
                        if (newCol >= BOARD_SIZE) newCol = 0;
                        if (newCol < 0) newCol = BOARD_SIZE - 1;
                        p.position.col = newCol;
                    }
                });
            } else { // Col
                const colIdx = target.col;
                const colArray = board.map(r => r[colIdx]);
                if (direction === 1) { // Down
                    const popped = colArray.pop();
                    colArray.unshift(popped);
                } else { // Up
                    const popped = colArray.shift();
                    colArray.push(popped);
                }
                // Write back
                for(let r=0; r<BOARD_SIZE; r++) {
                    board[r][colIdx] = colArray[r];
                }
                // Update player positions
                Object.values(players).forEach(p => {
                    if (p.position && p.position.col === colIdx) {
                        let newRow = p.position.row + direction;
                        if (newRow >= BOARD_SIZE) newRow = 0;
                        if (newRow < 0) newRow = BOARD_SIZE - 1;
                        p.position.row = newRow;
                    }
                });
            }
            success = true;
            message = 'Board shifted.';
        } else {
            message = 'Click an edge tile to shift that row/column.';
        }
        break;
    }

    if (success) {
        // Decrement actions
        newState.actionsRemaining--;
        refreshCapturedWells(newState);
        
        // Check immediate victory condition
        if (checkPathVictory(newState, newState.activePlayer)) {
            newState.winner = newState.activePlayer;
            newState.winReason = 'Path Completed!';
        }

        // If turn over
        if (!newState.winner && newState.actionsRemaining <= 0) {
            if (checkPowerWellVictory(newState, newState.activePlayer)) {
                newState.winner = newState.activePlayer;
                newState.winReason = '3 Power Wells Captured!';
            }

            if (newState.winner) {
                newState.selectedCardId = null;
                newState.selectedActionIndex = null;
                return { newState, success: true, message };
            }

            // Switch player
            newState.activePlayer = newState.activePlayer === 1 ? 2 : 1;
            newState.actionsRemaining = 2;
            newState.turn += 1;
            newState.selectedCardId = null;
            newState.selectedActionIndex = null;
            message += ` Turn over. Player ${newState.activePlayer}'s turn.`;
        } else {
            // Same player continues
            newState.selectedCardId = null;
            newState.selectedActionIndex = null;
        }
        return { newState, success: true, message };
    } else {
        return { newState: gameState, success: false, message };
    }
};

// Check if a move is valid
export const isValidMove = (board, from, to) => {
  if (!from || !to || typeof from.row !== 'number' || typeof from.col !== 'number' ||
      typeof to.row !== 'number' || typeof to.col !== 'number') return false;

  if (to.row < 0 || to.row >= BOARD_SIZE || to.col < 0 || to.col >= BOARD_SIZE) return false;
  
  const fromTile = board[from.row][from.col];
  const toTile = board[to.row][to.col];

  if (!fromTile || !toTile) return false;

  // Check connectivity
  // 1. Get connections of FromTile at its current rotation
  const fromConns = getRotatedConnections(fromTile.type, fromTile.rotation);
  // 2. Get connections of ToTile at its current rotation
  const toConns = getRotatedConnections(toTile.type, toTile.rotation);

  // Determine direction of movement
  const dRow = to.row - from.row;
  const dCol = to.col - from.col;

  // North (dRow -1)
  if (dRow === -1 && dCol === 0) {
    return fromConns[0] && toConns[2]; // From North connects to To South
  }
  // East (dCol 1)
  if (dRow === 0 && dCol === 1) {
    return fromConns[1] && toConns[3]; // From East connects to To West
  }
  // South (dRow 1)
  if (dRow === 1 && dCol === 0) {
    return fromConns[2] && toConns[0]; // From South connects to To North
  }
  // West (dCol -1)
  if (dRow === 0 && dCol === -1) {
    return fromConns[3] && toConns[1]; // From West connects to To East
  }

  return false;
};

// Get connections [N, E, S, W] after rotation
export const getRotatedConnections = (type, rotation) => {
  if (!TILE_CONNECTIONS) {
    console.error("TILE_CONNECTIONS undefined in getRotatedConnections");
    return [false, false, false, false];
  }
  const base = TILE_CONNECTIONS[type];
  if (!base) {
    console.error(`Unknown tile type: ${type}`);
    return [false, false, false, false];
  }
  // rotation 0: 0 shifts. 90: 1 shift right. 180: 2 shifts. 270: 3 shifts.
  const shifts = rotation / 90;
  
  // Rotate the array to the right by 'shifts'
  const rotated = [...base];
  for (let i = 0; i < shifts; i++) {
    const last = rotated.pop();
    rotated.unshift(last);
  }
  return rotated;
};

// BFS for path victory
export const checkPathVictory = (gameState, playerId) => {
  return getPathProgress(gameState, playerId) === 100; // 100 indicates reached goal
};

// Returns a score 0-100 indicating progress towards goal row
export const getPathProgress = (gameState, playerId) => {
  if (!gameState.players || !gameState.players[playerId]) return 0;
  const player = gameState.players[playerId];
  const startRow = player.homeRow;
  const goalRow = player.goalRow;

  const queue = [];
  const visited = new Set();
  let maxProgress = 0; // 0 to BOARD_SIZE-1

  // Start nodes
  for (let c = 0; c < BOARD_SIZE; c++) {
    if (gameState.board[startRow][c]) {
      queue.push({ pos: { row: startRow, col: c }, dist: 0 });
      visited.add(`${startRow},${c}`);
    }
  }

  while (queue.length > 0) {
    const { pos: curr, dist } = queue.shift();
    
    // Calculate progress based on row distance from start
    const currentProgress = Math.abs(curr.row - startRow);
    if (currentProgress > maxProgress) {
        maxProgress = currentProgress;
    }

    if (curr.row === goalRow) return 100; // Victory

    const neighbors = [
      { r: curr.row - 1, c: curr.col }, // N
      { r: curr.row, c: curr.col + 1 }, // E
      { r: curr.row + 1, c: curr.col }, // S
      { r: curr.row, c: curr.col - 1 }, // W
    ];

    for (const n of neighbors) {
      if (isValidMove(gameState.board, curr, n)) {
        const key = `${n.r},${n.c}`;
        if (!visited.has(key)) {
          visited.add(key);
          queue.push({ pos: { row: n.r, col: n.c }, dist: dist + 1 });
        }
      }
    }
  }

  // Normalize progress to score (excluding 100 which is win)
  // Max possible row diff is BOARD_SIZE - 1 (e.g. 4)
  // So score = (maxProgress / 4) * 90?
  return Math.min(99, (maxProgress / (BOARD_SIZE - 1)) * 90);
};

export const checkPowerWellVictory = (gameState, playerId) => {
  if (!gameState?.powerWells || !gameState?.board) return false;

  let lockedWellCount = 0;
  for (const well of gameState.powerWells) {
    const tile = gameState.board[well.row]?.[well.col];
    if (tile?.hasResonator && tile.resonatorOwner === playerId) {
      lockedWellCount++;
    }
  }

  return lockedWellCount >= 3;
};