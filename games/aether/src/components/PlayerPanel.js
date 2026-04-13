import React from 'react';
import { Box, Zap } from 'lucide-react';

const PlayerPanel = ({ player, isActive, capturedWells }) => {
  const isRed = player.color === 'red';
  const capturedCount = Object.values(capturedWells).filter((id) => id === player.id).length;

  return (
    <div className={`p-6 rounded-2xl border transition-all duration-300 ${isActive ? (isRed ? 'bg-red-950/30 border-red-500/50 shadow-[0_0_30px_rgba(239,68,68,0.1)]' : 'bg-blue-950/30 border-blue-500/50 shadow-[0_0_30px_rgba(59,130,246,0.1)]') : 'bg-neutral-900 border-neutral-800 opacity-70'}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className={`text-xl font-bold ${isRed ? 'text-red-400' : 'text-blue-400'}`}>{player.name}</h2>
        {isActive && <div className="w-2 h-2 rounded-full bg-white animate-pulse" />}
      </div>
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="text-neutral-500 text-xs font-mono uppercase">Resonators</span>
          <div className="flex gap-1">
            {Array.from({ length: 4 }).map((_, i) => <Box key={i} size={16} className={i < player.resonators ? (isRed ? 'text-red-500' : 'text-blue-500') : 'text-neutral-800'} fill={i < player.resonators ? 'currentColor' : 'none'} />)}
          </div>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-neutral-500 text-xs font-mono uppercase">Power Wells</span>
          <div className="flex gap-1">
            {Array.from({ length: 3 }).map((_, i) => <Zap key={i} size={16} className={i < capturedCount ? 'text-yellow-500' : 'text-neutral-800'} fill={i < capturedCount ? 'currentColor' : 'none'} />)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlayerPanel;