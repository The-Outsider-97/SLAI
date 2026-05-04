export const BOARD_SIZE = 5;
export const POWER_WELLS = [
  { row: 0, col: 0 },
  { row: 0, col: 4 },
  { row: 4, col: 0 },
  { row: 4, col: 4 },
];

export const TILE_TYPES = ['STRAIGHT', 'CURVE', 'T_JUNCTION'];

export const TILE_CONNECTIONS = {
  STRAIGHT: [true, false, true, false],
  CURVE: [true, true, false, false],
  T_JUNCTION: [false, true, true, true],
  CROSS: [true, true, true, true],
};

export const INITIAL_DECK_SIZE = 15;
export const ACTION_TYPES = ['PLACE', 'SHIFT', 'ROTATE', 'ADVANCE', 'ATTUNE'];

export const generateDeck = () => ACTION_TYPES.map((action) => ({
  id: `card-${action.toLowerCase()}`,
  actions: [action],
}));