import React, { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { createInitialState, executeAction } from './utils/gameLogic.js';
import { requestAetherMove } from './service/aetherAiClient.js';
import { computeMatchRecord, SCOREBOARD_STORAGE_KEY } from './utils/scoring.js';
import Board from './components/Board.js';
import ActionDeck from './components/ActionDeck.js';
import PlayerPanel from './components/PlayerPanel.js';
import SideBar from './components/SideBar.js';

const QUICK_GUIDE = [
  'Goal: Connect your edge (Red: Top, Blue: Bottom) to the opposite side OR end your round with 3 locked Power Wells.',
  'Turn: Pick 2 actions from the cards below.',
  'Shift: Click edge tiles to slide rows/cols.',
  'Attune: Lock your current tile & capture Wells (grey tiles cannot be used to capture Wells).',
  'Place: Add a tile to an empty slot.',
  'Rotate: Rotate a tile.',
  'Advance: Move character to the adjacent connected tile.',
];


const loadMatchHistory = () => {
  try {
    const raw = localStorage.getItem(SCOREBOARD_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

export default function App() {
  const [gameState, setGameState] = useState(() => createInitialState());
  const [activityLog, setActivityLog] = useState(['System ready.']);
  const [matchHistory, setMatchHistory] = useState(() => loadMatchHistory());
  const [matchLogged, setMatchLogged] = useState(false);

  const postSystemMessage = (message) => {
    if (!message) return;
    setActivityLog((prev) => [message, ...prev].slice(0, 20));
  };

  useEffect(() => {
    localStorage.setItem(SCOREBOARD_STORAGE_KEY, JSON.stringify(matchHistory));
  }, [matchHistory]);

  useEffect(() => {
    if (!gameState.winner || matchLogged) return;
    const record = computeMatchRecord(gameState);
    setMatchHistory((prev) => [record, ...prev].slice(0, 12));
    setMatchLogged(true);
  }, [gameState, matchLogged]);

  useEffect(() => {
    if (gameState.mode !== 'PVAI' || gameState.activePlayer !== 2 || gameState.winner) return;

    const timer = setTimeout(async () => {
      const move = await requestAetherMove(gameState);
      if (!move) {
        postSystemMessage('AI skipped turn.');
        setGameState((prev) => ({ ...prev, activePlayer: 1, actionsRemaining: 2 }));
        return;
      }

      setGameState((prev) => ({ ...prev, selectedCardId: move.cardId, selectedActionIndex: move.actionIndex }));
      setTimeout(() => {
        setGameState((prev) => {
          const card = prev.faceUpCards.find((c) => c.id === move.cardId);
          const action = card?.actions?.[move.actionIndex];
          if (!action) {
            postSystemMessage('AI selected an invalid action.');
            return prev;
          }

          const result = executeAction(prev, action, move.target);
          postSystemMessage(result.message || 'AI action resolved.');
          return result.success ? result.newState : prev;
        });
      }, 350);
    }, 800);

    return () => clearTimeout(timer);
  }, [gameState.activePlayer, gameState.actionsRemaining, gameState.mode, gameState.winner]);

  const handleCardSelect = (cardId, actionIndex) => {
    if (gameState.winner || (gameState.mode === 'PVAI' && gameState.activePlayer === 2)) return;
    setGameState((prev) => ({ ...prev, selectedCardId: cardId, selectedActionIndex: actionIndex }));
  };

  const handleTileClick = (row, col) => {
    if (gameState.winner || (gameState.mode === 'PVAI' && gameState.activePlayer === 2)) return;
    if (!gameState.selectedCardId || gameState.selectedActionIndex === null) return;

    const card = gameState.faceUpCards.find((c) => c.id === gameState.selectedCardId);
    const action = card?.actions?.[gameState.selectedActionIndex];
    if (action) {
      const result = executeAction(gameState, action, { row, col });
      if (result.success) setGameState(result.newState);
      postSystemMessage(result.message);
    }
  };

  const handleModeChange = (mode) => {
    const aiStarts = mode === 'PVAI';
    setGameState(createInitialState({ mode, aiStarts }));
    setMatchLogged(false);
    setActivityLog([`Mode switched: ${mode === 'PVAI' ? 'Player v. AI' : 'Player v. Player'}.`]);
  };

  const handleNewGame = () => {
    setGameState(createInitialState({ mode: gameState.mode, aiStarts: gameState.mode === 'PVAI' }));
    setMatchLogged(false);
    setActivityLog(['New game initialized.']);
  };

  const winnerName = gameState.winner ? gameState.players[gameState.winner].name : '';
  const currentMatchPreview = useMemo(() => computeMatchRecord(gameState), [gameState]);

  return (
    <div className="min-h-screen bg-neutral-900 text-neutral-200 font-sans selection:bg-indigo-500/30">
      <div className="mx-auto max-w-[1420px] px-3 py-4 lg:pr-[370px]">
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_390px] gap-6 items-start border-b border-white/10 pb-4">
          <section className="relative min-h-[760px]">
            <div className="flex items-center gap-3">
              <h1 className="text-[44px] font-extrabold tracking-[-0.03em] leading-none text-white">
                AETHER <span className="text-red-500">SHIFT</span>
              </h1>
              <button
                type="button"
                onClick={() => setGameState((prev) => ({ ...prev, showGuide: !prev.showGuide }))}
                className="inline-flex items-center gap-1 rounded-full border border-white/20 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-neutral-300 hover:text-white hover:border-indigo-400"
              >
                Quick Guide {gameState.showGuide ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
            </div>
            <p className="mt-2 text-sm tracking-[0.22em] text-neutral-500">TACTICAL SPATIAL WARFARE</p>

            {gameState.showGuide && (
              <div className="absolute top-[76px] left-0 right-0 z-20 rounded-xl border border-indigo-400/40 bg-indigo-950/35 p-4 text-sm">
                <p className="font-semibold text-indigo-100">Quick Guide</p>
                <ul className="mt-2 list-disc list-inside space-y-1 text-indigo-100/90">
                  {QUICK_GUIDE.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          <section className="space-y-4 w-[390px]">
            <div className="text-right pb-2">
              <div className={`text-[38px] font-bold leading-none ${gameState.activePlayer === 1 ? 'text-red-500' : 'text-blue-500'}`}>
                {gameState.players[gameState.activePlayer].name}'s Turn
              </div>
              <div className="text-xs uppercase tracking-[0.24em] text-neutral-300">Actions: {gameState.actionsRemaining}</div>
              <div className="mt-2 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => handleModeChange('PVAI')}
                  className={`rounded-lg border px-3 py-1 text-[11px] font-semibold ${
                    gameState.mode === 'PVAI' ? 'border-indigo-400 text-indigo-200' : 'border-white/20 text-neutral-400'
                  }`}
                >
                  Player v. AI
                </button>
                <button
                  type="button"
                  onClick={() => handleModeChange('PVP')}
                  className={`rounded-lg border px-3 py-1 text-[11px] font-semibold ${
                    gameState.mode === 'PVP' ? 'border-indigo-400 text-indigo-200' : 'border-white/20 text-neutral-400'
                  }`}
                >
                  Player v. Player
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <PlayerPanel player={gameState.players[1]} isActive={gameState.activePlayer === 1} capturedWells={gameState.capturedWells} />
              <PlayerPanel player={gameState.players[2]} isActive={gameState.activePlayer === 2} capturedWells={gameState.capturedWells} />
            </div>

            <div className="relative flex justify-center">
              <Board gameState={gameState} onTileClick={handleTileClick} />
            </div>

            <div className="space-y-2">
              <p className="text-xs uppercase tracking-[0.3em] text-neutral-500">Action Deck</p>
              <ActionDeck
                cards={gameState.faceUpCards}
                selectedCardId={gameState.selectedCardId}
                selectedActionIndex={gameState.selectedActionIndex}
                onSelect={handleCardSelect}
                disabled={!!gameState.winner || (gameState.mode === 'PVAI' && gameState.activePlayer === 2)}
              />
            </div>
          </section>
        </div>
      </div>

      {gameState.winner && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-xl px-4">
          <div className="relative w-full max-w-2xl overflow-hidden rounded-[2rem] border border-white/10 bg-neutral-950/90 p-10 shadow-2xl ring-1 ring-white/10">
            <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent pointer-events-none" />
            <div className="relative flex flex-col items-center gap-3 text-center">
              <h2 className={`text-7xl font-black tracking-tighter ${gameState.winner === 1 ? 'text-red-400' : 'text-blue-400'} drop-shadow-[0_0_20px_rgba(99,102,241,0.45)]`}>
                {gameState.winner === 1 ? 'VICTORY' : gameState.mode === 'PVAI' ? 'DEFEAT' : 'VICTORY'}
              </h2>
              <div className="text-[10px] font-bold text-white/50 tracking-[0.4em] uppercase">
                {gameState.winReason === 'Path Completed!' ? 'EDGE LINK ESTABLISHED' : 'WELL DOMINANCE ACHIEVED'}
              </div>
              <p className="text-sm text-neutral-300">{winnerName} wins the match.</p>
            </div>

            <div className="relative mt-8 rounded-2xl border border-white/10 bg-black/40 px-6 py-5 text-sm text-neutral-200">
              <div className="grid grid-cols-2 gap-y-2">
                <p>Mode: <span className="text-white">{currentMatchPreview.mode}</span></p>
                <p>Turns and Rounds: <span className="font-mono text-white">{currentMatchPreview.turns} / {currentMatchPreview.rounds}</span></p>
                <p>Wells -- Red: <span className="font-mono text-white">{currentMatchPreview.wells.red}</span> / Blue: <span className="font-mono text-white">{currentMatchPreview.wells.blue}</span></p>
                <p>Winner: <span className="text-white">{currentMatchPreview.winner}</span></p>
                <p>Score: <span className="font-mono text-white">{currentMatchPreview.score}</span></p>
                <p>Points: <span className={`font-mono ${currentMatchPreview.points > 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{currentMatchPreview.points}</span></p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleNewGame}
              className="relative mt-8 w-full rounded-2xl bg-white py-4 text-xs font-black uppercase tracking-[0.3em] text-black transition-all hover:bg-indigo-400 hover:text-white"
            >
              Start New Match
            </button>
          </div>
        </div>
      )}

      <SideBar
        className="fixed right-0 top-24 h-[78vh] z-30"
        title="Scoreboard"
        commsTitle="Comms"
        scoreboardContent={(
          <div className="space-y-3 text-sm">
            {matchHistory.length === 0 ? (
              <div className="rounded-xl border border-white/10 bg-black/30 p-3 text-xs text-neutral-400">No completed matches yet.</div>
            ) : (
              matchHistory.map((entry) => (
                <div key={entry.id} className="rounded-xl border border-white/10 bg-black/30 p-3 space-y-1 text-[12px]">
                  <div className="text-[10px] uppercase tracking-widest text-neutral-500">{entry.timestamp}</div>
                  <div>Mode: <span className="text-neutral-200">{entry.mode}</span></div>
                  <div>Turns and Rounds: <span className="font-mono text-neutral-200">{entry.turns} / {entry.rounds}</span></div>
                  <div>Wells -- Red: <span className="font-mono text-neutral-200">{entry.wells.red}</span> / Blue: <span className="font-mono text-neutral-200">{entry.wells.blue}</span></div>
                  <div>Winner: <span className="text-neutral-200">{entry.winner}</span></div>
                  <div>Score: <span className="font-mono text-neutral-200">{entry.score}</span></div>
                  <div>Points: <span className={`font-mono ${entry.points > 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{entry.points}</span></div>
                </div>
              ))
            )}
          </div>
        )}
        commsContent={(
          <div className="rounded-xl border border-white/10 bg-black/30 p-3 h-[64vh] overflow-y-auto text-xs font-mono text-neutral-300 space-y-2">
            {activityLog.map((entry, index) => (
              <p key={`${entry}-${index}`} className="border-b border-white/5 pb-2">{entry}</p>
            ))}
          </div>
        )}
      />
    </div>
  );
}