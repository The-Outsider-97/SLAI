export const TRACK_LENGTH = 10;
export const TOKEN_COUNT = 5;
export const SAFE_CELLS = new Set([0, TRACK_LENGTH - 1]);

function createCell(index) {
  const cell = document.createElement('button');
  cell.type = 'button';
  cell.className = 'cell';
  cell.dataset.index = String(index);
  if (SAFE_CELLS.has(index)) {
    cell.classList.add('safe');
  }
  return cell;
}

export function createBoard(boardEl, onCellClick) {
  boardEl.innerHTML = '';
  const cells =[];

  for (let i = 0; i < TRACK_LENGTH; i += 1) {
    const cell = createCell(i);
    cell.addEventListener('click', () => onCellClick(i));
    cells.push(cell);
    boardEl.appendChild(cell);
  }

  return cells;
}

export function renderBoard(cells, state) {
  // 1. Record old positions for FLIP animation
  const oldPositions = new Map();
  document.querySelectorAll('.disc').forEach(disc => {
    if (disc.dataset.tokenId) {
      oldPositions.set(disc.dataset.tokenId, disc.getBoundingClientRect());
    }
  });

  // 2. Render new state
  cells.forEach((cell) => {
    cell.innerHTML = '';
    const idx = Number(cell.dataset.index);
    for (const color of ['light', 'dark']) {
      const tokens = state[color].tokens.filter((token) => token.position === idx);
      tokens.forEach((token) => {
        const disc = document.createElement('div');
        disc.className = `disc ${color}`;
        disc.dataset.tokenId = `${color}-${token.id}`;
        cell.appendChild(disc);
      });
    }
  });

  // 3. Measure new positions and animate
  document.querySelectorAll('.disc').forEach(disc => {
    const id = disc.dataset.tokenId;
    if (oldPositions.has(id)) {
      const oldRect = oldPositions.get(id);
      const newRect = disc.getBoundingClientRect();

      const deltaX = oldRect.left - newRect.left;
      const deltaY = oldRect.top - newRect.top;

      if (deltaX !== 0 || deltaY !== 0) {
        disc.animate([
          { transform: `translate(${deltaX}px, ${deltaY}px)` },
          { transform: 'translate(0, 0)' }
        ], {
          duration: 400,
          easing: 'ease-out'
        });
      }
    } else {
      // New token appearing on board (fade/scale in)
      disc.animate([
        { transform: 'scale(0.5)', opacity: 0 },
        { transform: 'scale(1)', opacity: 1 }
      ], {
        duration: 300,
        easing: 'ease-out'
      });
    }
  });
}

export function throwSticks(stickCount = 4) {
  const sticks = Array.from({ length: stickCount }, () => Math.random() < 0.5);
  const marked = sticks.filter(Boolean).length;
  const moveDistance = marked === 0 ? 5 : marked;
  return { sticks, moveDistance };
}

export function renderThrow(throwTrayEl, sticks) {
  throwTrayEl.innerHTML = '';
  sticks.forEach((marked) => {
    const stick = document.createElement('div');
    stick.className = `stick ${marked ? 'marked' : ''}`;
    throwTrayEl.appendChild(stick);
  });
}