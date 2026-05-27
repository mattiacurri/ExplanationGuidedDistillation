import { useApp } from '../store';

export function TopBar() {
  const { state, setView } = useApp();

  return (
    <header className="flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-accent text-white font-extrabold text-sm select-none">
          CV
        </span>
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-slate-900 leading-none">
            Arena
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">{state.statusText}</p>
        </div>
      </div>

      <div className="flex bg-slate-100 rounded-lg p-0.5">
        <button
          onClick={() => setView('arena')}
          className={`px-4 py-1.5 text-sm font-semibold rounded-md transition-colors ${
            state.view === 'arena'
              ? 'bg-white text-accent shadow-sm'
              : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          Arena
        </button>
        <button
          onClick={() => setView('dashboard')}
          className={`px-4 py-1.5 text-sm font-semibold rounded-md transition-colors ${
            state.view === 'dashboard'
              ? 'bg-white text-accent shadow-sm'
              : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          Dashboard
        </button>
      </div>
    </header>
  );
}
