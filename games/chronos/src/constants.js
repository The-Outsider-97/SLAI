export const BOARD_VARIANTS = [9, 11, 13];

export const CONFIG = {
  board: {
    size: 9,
    core: {
      rows: [3, 5],
      cols: [3, 5]
    }
  },
  home_rows: [0, 8],
  unit_types: {
    Scout: {
      symbol: 'S',
      movement: 2,
      health: 1,
      is_commander: false,
      count_per_player: 6,
      icon: 'running',
      value: 1
    },
    Warden: {
      symbol: 'W',
      movement: 1,
      health: 2,
      is_commander: false,
      count_per_player: 2,
      icon: 'shield',
      value: 2
    },
    Strategos: {
      symbol: 'C',
      movement: 2,
      health: 1,
      is_commander: true,
      count_per_player: 1,
      icon: 'crown',
      value: 3
    }
  },
  action_tokens: [1, 2, 3, 4, 5],
  attack_threshold: 3
};

export function applyBoardSize(boardSize) {
  const size = Number(boardSize);
  if (!BOARD_VARIANTS.includes(size)) return;

  CONFIG.board.size = size;
  const center = Math.floor(size / 2);
  CONFIG.board.core.rows = [center - 1, center + 1];
  CONFIG.board.core.cols = [center - 1, center + 1];
  CONFIG.home_rows = [0, size - 1];
}

export const ActionType = {
  MOVE: 'move',
  ATTACK: 'attack',
  CLAIM: 'claim',
  PASS: 'pass'
};
