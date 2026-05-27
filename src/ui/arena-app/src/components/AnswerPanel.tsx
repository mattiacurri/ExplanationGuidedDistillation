import { useApp } from '../store';
import type { VoteChoice } from '../types';

interface Props {
  side: 'left' | 'right';
  onVote: (c: VoteChoice) => void;
}

const BADGE = { left: 'A', right: 'B' } as const;

export function AnswerPanel({ side, onVote }: Props) {
  const { state } = useApp();
  if (!state.sample) return null;

  const comp = state.sample.comparison[side];
  const rev = state.sample.reveal[side];
  const revealed = state.revealedChoice !== null;
  const chosen = state.revealedChoice === side;
  const badge = BADGE[side];

  const label = revealed ? `Risposta ${badge} — ${rev.label}` : comp.blind_name;
  const btnLabel = revealed
    ? chosen
      ? 'Scelta salvata'
      : `${badge} rivelata`
    : `Preferisco ${badge}`;

  return (
    <div
      className={`flex-1 min-h-0 flex flex-col bg-white rounded-xl shadow-md border transition-all ${
        chosen
          ? 'border-accent ring-2 ring-accent/30'
          : 'border-slate-200'
      }`}
    >
      {/* Header */}
      <div className={`flex items-center gap-3 px-4 py-2.5 border-b border-slate-200 flex-shrink-0 rounded-t-xl ${
        revealed ? 'bg-slate-50' : ''
      }`}>
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-accent text-white text-xs font-bold flex-shrink-0">
          {badge}
        </span>
        <span className="font-bold text-sm text-slate-800 truncate">{label}</span>
      </div>

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-auto p-4">
        <p className="text-[15px] leading-relaxed text-slate-700 whitespace-pre-wrap">
          {comp.text}
        </p>
      </div>

      {/* Footer */}
      <div className="border-t border-slate-200 px-3 py-2.5 flex-shrink-0">
        <button
          onClick={() => onVote(side)}
          disabled={state.busy || revealed}
          className="w-full py-2 text-sm font-semibold rounded-lg bg-accent text-white hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {btnLabel}
        </button>
      </div>
    </div>
  );
}
