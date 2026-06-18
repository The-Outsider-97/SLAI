import { CONFIG, ActionType } from './constants.js?v=chronos-ui-v4-20260617-2208';

const DIRECTIONS = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1],           [0, 1],
  [1, -1],  [1, 0],  [1, 1],
];

function clampBoard(value) {
  return Number.isInteger(value) && value >= 0 && value < CONFIG.board.size;
}

function sameCell(a, b) {
  return a && b && a.row === b.row && a.col === b.col;
}

export class Unit {
  constructor({ id, type, playerId, row, col }) {
    const props = CONFIG.unit_types[type] || CONFIG.unit_types.Scout;
    this.id = id;
    this.type = type;
    this.playerId = playerId;
    this.row = row;
    this.col = col;
    this.props = props;
    this.maxHealth = props.health;
    this.health = props.health;
    this.lastActedRound = 0;
  }

  canAct(round) {
    return this.health > 0 && this.lastActedRound !== round;
  }

  toJSON() {
    return {
      id: this.id,
      unitId: this.id,
      type: this.type,
      owner: this.playerId,
      playerId: this.playerId,
      row: this.row,
      col: this.col,
      r: this.row,
      c: this.col,
      hp: this.health,
      health: this.health,
      maxHealth: this.maxHealth,
      value: this.props.value,
    };
  }
}

export class Player {
  constructor(id) {
    this.id = id;
    this.score = 0;
    this.tokens = [...CONFIG.action_tokens];
    this.usedTokens = new Set();
    this.units = [];
  }

  getActiveUnits(round) {
    return this.units.filter((unit) => unit.canAct(round));
  }

  resetRound() {
    this.usedTokens.clear();
  }
}

export class Board {
  constructor() {
    this.size = CONFIG.board.size;
  }

  isCoreCell(row, col) {
    const [minRow, maxRow] = CONFIG.board.core.rows;
    const [minCol, maxCol] = CONFIG.board.core.cols;
    return row >= minRow && row <= maxRow && col >= minCol && col <= maxCol;
  }

  getUnitAt(row, col, players = null) {
    const sourcePlayers = players || this.players || [];
    for (const player of sourcePlayers) {
      const unit = player.units.find((candidate) => (
        candidate.health > 0 && candidate.row === row && candidate.col === col
      ));
      if (unit) return unit;
    }
    return null;
  }
}

export class Game {
  constructor() {
    this.board = new Board();
    this.players = [new Player(0), new Player(1)];
    this.board.players = this.players;
    this.round = 1;
    this.turnCount = 0;
    this.phase = 'planning';
    this.currentPlayerId = 0;
    this.currentSlot = 0;
    this.timeline = [];
    this.requiredActionsPerPlayer = [0, 0];
    this.winner = null;
    this.winningEvent = null;
    this.strategosChoices = new Map();

    this.setupUnits();
    this.startPlanningRound();
  }

  setupUnits() {
    const size = CONFIG.board.size;
    const p2Row = 0;
    const p1Row = size - 1;
    const unitOrder = ['Scout', 'Warden', 'Scout', 'Scout', 'Strategos', 'Scout', 'Scout', 'Warden', 'Scout'];
    const startCol = Math.max(0, Math.floor((size - unitOrder.length) / 2));

    for (let playerId = 0; playerId <= 1; playerId += 1) {
      const row = playerId === 0 ? p1Row : p2Row;
      const units = unitOrder.map((type, index) => new Unit({
        id: `p${playerId}-${type.toLowerCase()}-${index}`,
        type,
        playerId,
        row,
        col: startCol + index,
      }));
      this.players[playerId].units = units;
    }
  }

  startPlanningRound() {
    this.phase = 'planning';
    this.currentPlayerId = 0;
    this.currentSlot = 0;
    this.players.forEach((player) => player.resetRound());
    this.requiredActionsPerPlayer = this.players.map((player) => Math.min(CONFIG.action_tokens.length, player.getActiveUnits(this.round).length));
    const slots = Math.max(1, ...this.requiredActionsPerPlayer);
    this.timeline = Array.from({ length: slots }, () => [null, null]);
  }

  getCorePoints(playerId) {
    return this.players[playerId].units.reduce((total, unit) => {
      if (unit.health <= 0 || !this.board.isCoreCell(unit.row, unit.col)) return total;
      return total + (unit.props.value || 1);
    }, 0);
  }

  getPossibleActions(unit) {
    if (!unit || unit.health <= 0) return { move: [], attack: [], claim: false };

    const move = [];
    const attack = [];
    const movement = Number(unit.props.movement || 1);

    for (const [dr, dc] of DIRECTIONS) {
      for (let step = 1; step <= movement; step += 1) {
        const row = unit.row + dr * step;
        const col = unit.col + dc * step;
        if (!clampBoard(row) || !clampBoard(col)) break;

        const occupant = this.board.getUnitAt(row, col, this.players);
        if (occupant) {
          if (occupant.playerId !== unit.playerId) attack.push(occupant);
          break;
        }
        move.push({ r: row, c: col });
      }
    }

    return {
      move,
      attack,
      claim: this.board.isCoreCell(unit.row, unit.col),
    };
  }

  placeAction(playerId, slot, token, unit, type, params = {}) {
    if (this.phase !== 'planning') return false;
    if (playerId !== this.currentPlayerId) return false;
    if (slot !== this.currentSlot || !this.timeline[slot]) return false;

    const player = this.players[playerId];
    const actionToken = Number(token);
    if (!player.tokens.includes(actionToken) || player.usedTokens.has(actionToken)) return false;

    const action = this.createAction(playerId, actionToken, unit, type, params);
    if (!action || !this.isActionLegal(action)) return false;

    this.timeline[slot][playerId] = action;
    player.usedTokens.add(actionToken);
    if (action.unit) action.unit.lastActedRound = this.round;
    this.advancePlanningTurn();
    return true;
  }

  createAction(playerId, token, unit, type, params = {}) {
    if (type === ActionType.PASS) {
      return { type: ActionType.PASS, token, playerId, unit: null, unitId: null, target: null, params: {} };
    }

    if (!unit || unit.playerId !== playerId || !unit.canAct(this.round)) return null;

    if (type === ActionType.MOVE) {
      const target = params.target;
      if (!target) return null;
      return { type, token, playerId, unit, unitId: unit.id, target: { r: target.r, c: target.c }, params: { target: { r: target.r, c: target.c } } };
    }

    if (type === ActionType.ATTACK) {
      const target = params.target;
      if (!target) return null;
      return {
        type,
        token,
        playerId,
        unit,
        unitId: unit.id,
        target: this.serializeUnitTarget(target),
        params: { target },
      };
    }

    if (type === ActionType.CLAIM) {
      return { type, token, playerId, unit, unitId: unit.id, target: { r: unit.row, c: unit.col }, params: {} };
    }

    return null;
  }

  isActionLegal(action) {
    if (!action || typeof action !== 'object') return false;
    if (action.type === ActionType.PASS) return true;
    if (!action.unit || action.unit.health <= 0) return false;

    const actions = this.getPossibleActions(action.unit);
    if (action.type === ActionType.MOVE) {
      return actions.move.some((move) => move.r === action.target?.r && move.c === action.target?.c);
    }
    if (action.type === ActionType.ATTACK) {
      return actions.attack.some((target) => target.id === action.target?.id || sameCell(target, action.target));
    }
    if (action.type === ActionType.CLAIM) {
      return Boolean(actions.claim);
    }
    return false;
  }

  advancePlanningTurn() {
    if (this.currentPlayerId === 0) {
      this.currentPlayerId = 1;
      return;
    }

    this.currentSlot += 1;
    if (this.currentSlot >= this.timeline.length) {
      this.phase = 'resolution';
      this.currentSlot = 0;
      this.currentPlayerId = 0;
    } else {
      this.currentPlayerId = 0;
    }
  }

  resolveNextSlot() {
    if (this.phase !== 'resolution' || !this.timeline[this.currentSlot]) return null;

    const slot = this.timeline[this.currentSlot];
    const actions = slot
      .filter(Boolean)
      .sort((a, b) => (b.token - a.token) || (a.playerId - b.playerId));

    const results = [];
    for (const action of actions) {
      const result = this.resolveAction(action);
      if (result) results.push(result);
      if (this.phase === 'game_over' || this.phase === 'strategos_decision') break;
    }

    if (this.phase === 'resolution') {
      this.currentSlot += 1;
      this.evaluateVictory();
    }

    if (this.phase === 'resolution' && this.currentSlot >= this.timeline.length) {
      this.round += 1;
      this.turnCount += 1;
      this.evaluateVictory();
      if (this.phase === 'resolution') this.startPlanningRound();
    }

    return { results };
  }

  resolveAction(action) {
    if (!action) return null;
    if (action.type === ActionType.PASS) return `Player ${action.playerId + 1} passed with token ${action.token}.`;
    if (!action.unit || action.unit.health <= 0) return `Player ${action.playerId + 1}'s action fizzled.`;

    if (action.type === ActionType.CLAIM) {
      return this.board.isCoreCell(action.unit.row, action.unit.col)
        ? `${action.unit.type} claimed the core.`
        : `${action.unit.type} could not claim outside the core.`;
    }

    if (action.type === ActionType.MOVE) {
      const target = action.target;
      if (!target || !this.isEmpty(target.r, target.c)) return `${action.unit.type} could not move.`;
      action.unit.row = target.r;
      action.unit.col = target.c;
      return `${action.unit.type} moved to ${this.formatCell(target.r, target.c)}.`;
    }

    if (action.type === ActionType.ATTACK) {
      const target = this.findUnitById(action.target?.id) || this.board.getUnitAt(action.target?.row, action.target?.col, this.players);
      if (!target || target.health <= 0 || target.playerId === action.playerId) return `${action.unit.type} found no valid target.`;
      target.health = Math.max(0, target.health - 1);
      if (target.health === 0) {
        this.winningEvent = { type: 'assassination', victim: target, attacker: action.unit, player: action.playerId };
        return `${action.unit.type} eliminated ${target.type}.`;
      }
      return `${action.unit.type} damaged ${target.type}.`;
    }

    return null;
  }

  evaluateVictory() {
    const p1StrategosAlive = this.hasLivingStrategos(0);
    const p2StrategosAlive = this.hasLivingStrategos(1);
    const p1Core = this.getCorePoints(0);
    const p2Core = this.getCorePoints(1);

    if (p1Core >= 5 || p2Core >= 5) {
      if (p1Core === p2Core) return;
      this.winner = p1Core > p2Core ? 0 : 1;
      this.winningEvent = this.winningEvent || { type: 'core_control', player: this.winner };
      this.phase = 'game_over';
      return;
    }

    if (!p1StrategosAlive && !p2StrategosAlive) {
      this.phase = 'strategos_decision';
      this.currentPlayerId = 0;
      this.strategosChoices.clear();
      return;
    }

    if (!p1StrategosAlive || !p2StrategosAlive) {
      this.winner = p1StrategosAlive ? 0 : 1;
      this.phase = 'game_over';
    }
  }

  hasLivingStrategos(playerId) {
    return this.players[playerId].units.some((unit) => unit.type === 'Strategos' && unit.health > 0);
  }

  submitMutualStrategosChoice(playerId, choice) {
    if (this.phase !== 'strategos_decision') return false;
    const cleanChoice = choice === 'end' ? 'end' : 'continue';
    this.strategosChoices.set(playerId, cleanChoice);

    if (this.strategosChoices.size < 2) return true;

    const p1 = this.strategosChoices.get(0);
    const p2 = this.strategosChoices.get(1);
    if (p1 === 'continue' && p2 === 'continue') {
      this.round += 1;
      this.turnCount += 1;
      this.startPlanningRound();
    } else {
      this.winner = -1;
      this.phase = 'game_over';
    }
    return true;
  }

  executeRemoteMove(move) {
    if (!move || typeof move !== 'object') return false;
    const token = this.resolveRemoteToken(move.token);
    if (token === null) return false;

    const type = String(move.type || '').toLowerCase();
    if (type === ActionType.PASS) {
      return this.placeAction(1, this.currentSlot, token, null, ActionType.PASS, {});
    }

    const unit = this.findUnitById(move.unitId || move.unit_id);
    if (!unit || unit.playerId !== 1) return false;
    const target = move.target || move.params?.target || {};

    if (type === ActionType.MOVE) {
      return this.placeAction(1, this.currentSlot, token, unit, ActionType.MOVE, { target: { r: target.r ?? target.row, c: target.c ?? target.col } });
    }
    if (type === ActionType.ATTACK) {
      const victim = this.findUnitById(target.id) || this.board.getUnitAt(target.r ?? target.row, target.c ?? target.col, this.players);
      return this.placeAction(1, this.currentSlot, token, unit, ActionType.ATTACK, { target: victim || target });
    }
    if (type === ActionType.CLAIM) {
      return this.placeAction(1, this.currentSlot, token, unit, ActionType.CLAIM, {});
    }
    return false;
  }

  aiMove(playerId = 1) {
    const legalActions = this.getLegalActions(playerId);
    const preferred = legalActions.find((action) => action.type === ActionType.ATTACK)
      || legalActions.find((action) => action.type === ActionType.CLAIM)
      || legalActions.find((action) => action.type === ActionType.MOVE)
      || legalActions.find((action) => action.type === ActionType.PASS);

    if (!preferred) return false;
    if (preferred.type === ActionType.PASS) return this.placeAction(playerId, this.currentSlot, preferred.token, null, ActionType.PASS, {});

    const unit = this.findUnitById(preferred.unitId);
    if (!unit) return false;
    if (preferred.type === ActionType.MOVE) return this.placeAction(playerId, this.currentSlot, preferred.token, unit, ActionType.MOVE, { target: preferred.target });
    if (preferred.type === ActionType.ATTACK) {
      const target = this.findUnitById(preferred.target?.id);
      return this.placeAction(playerId, this.currentSlot, preferred.token, unit, ActionType.ATTACK, { target });
    }
    if (preferred.type === ActionType.CLAIM) return this.placeAction(playerId, this.currentSlot, preferred.token, unit, ActionType.CLAIM, {});
    return false;
  }

  getLegalActions(playerId = this.currentPlayerId) {
    const player = this.players[playerId];
    const token = this.firstAvailableToken(playerId);
    if (token === null) return [];

    const actedUnits = new Set();
    this.timeline.forEach((slot) => {
      slot.forEach((action) => {
        if (action?.unit?.playerId === playerId) actedUnits.add(action.unit.id);
      });
    });

    const actions = [];
    for (const unit of player.units) {
      if (!unit.canAct(this.round) || actedUnits.has(unit.id)) continue;
      const possible = this.getPossibleActions(unit);
      possible.attack.forEach((target) => actions.push(this.toRemoteAction(ActionType.ATTACK, token, unit, this.serializeUnitTarget(target))));
      if (possible.claim) actions.push(this.toRemoteAction(ActionType.CLAIM, token, unit, { r: unit.row, c: unit.col }));
      possible.move.forEach((target) => actions.push(this.toRemoteAction(ActionType.MOVE, token, unit, target)));
    }

    if (!actions.length || this.requiredActionsPerPlayer[playerId] >= CONFIG.action_tokens.length) {
      actions.push({ type: ActionType.PASS, token, playerId, unitId: null, target: null });
    }
    return actions;
  }

  serialize() {
    return {
      game: 'chronos',
      board_size: CONFIG.board.size,
      boardSize: CONFIG.board.size,
      phase: this.phase,
      round: this.round,
      turnNumber: this.turnCount,
      currentPlayer: this.currentPlayerId,
      currentPlayerId: this.currentPlayerId,
      currentSlot: this.currentSlot,
      players: this.players.map((player) => ({
        id: player.id,
        playerId: player.id,
        score: this.getCorePoints(player.id),
        tokens: player.tokens.filter((token) => !player.usedTokens.has(token)),
        usedTokens: [...player.usedTokens],
      })),
      units: this.players.flatMap((player) => player.units.map((unit) => unit.toJSON())),
      validMoves: this.phase === 'planning' ? this.getLegalActions(this.currentPlayerId) : [],
      timeline: this.timeline.map((slot) => slot.map((action) => action ? this.toSerializableAction(action) : null)),
    };
  }

  toRemoteAction(type, token, unit, target) {
    return {
      type,
      token,
      playerId: unit.playerId,
      unitId: unit.id,
      target: target || { r: unit.row, c: unit.col },
    };
  }

  toSerializableAction(action) {
    return {
      type: action.type,
      token: action.token,
      playerId: action.playerId,
      unitId: action.unitId,
      target: action.target,
    };
  }

  serializeUnitTarget(unit) {
    return {
      id: unit.id,
      type: unit.type,
      owner: unit.playerId,
      playerId: unit.playerId,
      row: unit.row,
      col: unit.col,
      r: unit.row,
      c: unit.col,
      hp: unit.health,
      health: unit.health,
    };
  }

  firstAvailableToken(playerId) {
    const player = this.players[playerId];
    const token = player.tokens.find((candidate) => !player.usedTokens.has(candidate));
    return Number.isFinite(token) ? token : null;
  }

  resolveRemoteToken(token) {
    const requested = Number(token);
    const player = this.players[1];
    if (player.tokens.includes(requested) && !player.usedTokens.has(requested)) return requested;
    return this.firstAvailableToken(1);
  }

  findUnitById(unitId) {
    if (!unitId) return null;
    return this.players.flatMap((player) => player.units).find((unit) => unit.id === unitId) || null;
  }

  isEmpty(row, col) {
    return clampBoard(row) && clampBoard(col) && !this.board.getUnitAt(row, col, this.players);
  }

  formatCell(row, col) {
    return `${String.fromCharCode(65 + col)}${CONFIG.board.size - row}`;
  }
}
