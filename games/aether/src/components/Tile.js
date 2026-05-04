import React from 'react';
import { motion } from 'motion/react';
import { TILE_CONNECTIONS } from '../constants.js';
import { User, Box, Zap } from 'lucide-react';

const Tile = ({ tile, onClick, isPowerWell, capturedBy }) => {
  const renderBasePaths = () => {
    if (!tile) return null;
    const [n, e, s, w] = TILE_CONNECTIONS[tile.type] || [false, false, false, false];
    const pathColor = tile.color === 'red' ? 'bg-red-500' : tile.color === 'blue' ? 'bg-blue-500' : 'bg-neutral-400';
    return (
      <motion.div className="absolute inset-0 flex items-center justify-center pointer-events-none" animate={{ rotate: tile.rotation || 0 }}>
        <div className={`w-4 h-4 rounded-full ${pathColor} z-10`} />
        {n && <div className={`absolute top-0 w-2 h-[50%] ${pathColor}`} />}
        {s && <div className={`absolute bottom-0 w-2 h-[50%] ${pathColor}`} />}
        {w && <div className={`absolute left-0 h-2 w-[50%] ${pathColor}`} />}
        {e && <div className={`absolute right-0 h-2 w-[50%] ${pathColor}`} />}
      </motion.div>
    );
  };

  return (
    <motion.div layout className={`relative w-16 h-16 md:w-20 md:h-20 rounded-lg cursor-pointer flex items-center justify-center ${tile ? 'bg-neutral-800' : 'bg-neutral-900 border-2 border-dashed border-neutral-800'} ${isPowerWell ? 'ring-2 ring-yellow-500/30' : ''}`} onClick={onClick}>
      {isPowerWell && <div className="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none"><Zap className={`w-12 h-12 ${capturedBy ? (capturedBy === 1 ? 'text-red-500' : 'text-blue-500') : 'text-yellow-500'}`} /></div>}
      {renderBasePaths()}
      {tile?.hasResonator && <div className="absolute top-1 right-1 z-20"><Box className={`w-4 h-4 ${tile.resonatorOwner === 1 ? 'text-red-400 fill-red-400' : 'text-blue-400 fill-blue-400'}`} /></div>}
      {tile?.playersPresent?.length > 0 && <div className="absolute z-30 flex gap-1">{tile.playersPresent.map((pid) => <div key={pid} className={`w-6 h-6 rounded-full flex items-center justify-center border-2 border-white ${pid === 1 ? 'bg-red-600' : 'bg-blue-600'}`}><User size={14} className="text-white" /></div>)}</div>}
    </motion.div>
  );
};

export default Tile;