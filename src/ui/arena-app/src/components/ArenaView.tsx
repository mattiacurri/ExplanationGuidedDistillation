import { useApp } from '../store';
import { useArena } from '../hooks/useArena';
import { AnswerPanel } from './AnswerPanel';
import type { VoteChoice } from '../types';

export function ArenaView() {
  const { state } = useApp();
  const { saveVote, fetchSample } = useArena();

  const handleVote = (choice: VoteChoice) => saveVote(choice);
  const revealed = state.revealedChoice !== null;
  const loading = state.busy || !state.sample;

  return (
    <div className="h-full flex flex-col gap-3">
      {/* Loading bar */}
      {loading && (
        <div className="flex-shrink-0 h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div className="h-full bg-accent rounded-full animate-progress" style={{ width: '40%' }} />
        </div>
      )}

      {/* Main content: image + answers */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(280px,1fr)_1.2fr] gap-3">
        {/* Image */}
        <div className="bg-white rounded-xl shadow-md border border-slate-200 overflow-hidden flex items-center justify-center">
          {state.sample ? (
            <img
              src={`${state.sample.sample.image_url}?t=${Date.now()}`}
              alt="Immagine del test set"
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="text-slate-300 text-sm">In attesa...</div>
          )}
        </div>

        {/* Answer panels */}
        <div className="flex flex-col gap-3 min-h-0">
          {state.sample ? (
            <>
              <AnswerPanel side="left" onVote={handleVote} />
              <AnswerPanel side="right" onVote={handleVote} />
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-300 text-sm">
              Caricamento confronto...
            </div>
          )}
        </div>
      </div>

      {/* Decision row */}
      <div className="flex-shrink-0 bg-white rounded-xl shadow-md border border-slate-200 px-4 py-3 flex items-center justify-center gap-3">
        <button
          onClick={() => handleVote('tie')}
          disabled={state.busy || revealed || !state.sample}
          className="px-5 py-2 text-sm font-semibold rounded-lg border-2 border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Pareggio
        </button>
        <button
          onClick={fetchSample}
          disabled={state.busy}
          className="px-5 py-2 text-sm font-semibold rounded-lg border-2 border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Salta match
        </button>
        {revealed && (
          <button
            onClick={fetchSample}
            disabled={state.busy}
            className="px-5 py-2 text-sm font-semibold rounded-lg bg-accent text-white hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Avanti
          </button>
        )}
      </div>
    </div>
  );
}
