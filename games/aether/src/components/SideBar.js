import React, { useState } from 'react';

export default function SideBar({
  title = 'Scoreboard',
  commsTitle = 'Comms',
  scoreboardContent,
  commsContent,
  defaultTab = 'scoreboard',
  className = '',
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <aside
      className={`border border-white/10 bg-neutral-950/95 rounded-l-2xl transition-all duration-200 ${
        collapsed ? 'w-14' : 'w-[320px]'
      } ${className}`}
    >
      <div className="flex items-center gap-2 border-b border-white/10 px-3 py-3">
        <button
          type="button"
          onClick={() => setCollapsed((value) => !value)}
          className="h-8 w-8 rounded-lg border border-white/10 text-xs font-bold text-neutral-300 hover:border-indigo-400 hover:text-white"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? '›' : '‹'}
        </button>
        {!collapsed && (
          <>
            <button
              type="button"
              onClick={() => setActiveTab('comms')}
              className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wider ${
                activeTab === 'comms' ? 'bg-indigo-500 text-white' : 'text-neutral-400'
              }`}
            >
              {commsTitle}
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('scoreboard')}
              className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wider ${
                activeTab === 'scoreboard' ? 'bg-indigo-500 text-white' : 'text-neutral-400'
              }`}
            >
              {title}
            </button>
          </>
        )}
      </div>

      {!collapsed && <div className="p-3">{activeTab === 'scoreboard' ? scoreboardContent : commsContent}</div>}
    </aside>
  );
}