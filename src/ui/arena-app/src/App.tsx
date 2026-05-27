import { useEffect } from 'react';
import { useApp } from './store';
import { useArena } from './hooks/useArena';
import { TopBar } from './components/TopBar';
import { ArenaView } from './components/ArenaView';
import { DashboardView } from './components/DashboardView';

export function App() {
  const { state } = useApp();
  const { fetchSources, fetchSample } = useArena();

  useEffect(() => {
    fetchSources().then(() => fetchSample());
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <TopBar />
      <main className="flex-1 min-h-0 p-3 md:p-4">
        {state.view === 'arena' ? <ArenaView /> : <DashboardView />}
      </main>
    </div>
  );
}
