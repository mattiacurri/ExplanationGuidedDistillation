import { useMemo } from 'react';
import { useApp } from '../store';
import { buildSummary, winRate } from '../hooks/useArena';

export function DashboardView() {
  const { state } = useApp();
  const sum = useMemo(() => buildSummary(state.sources, state.votes), [state.sources, state.votes]);

  return (
    <div className="h-full flex flex-col gap-3 overflow-auto">
      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricCard label="Voti totali" value={String(sum.totalVotes)} />
        <MetricCard
          label="Preferito"
          value={sum.topWinner ? `${sum.topWinner.label} (${sum.topWinner.wins})` : '—'}
        />
        <MetricCard label="Pareggi" value={String(sum.tieVotes)} />
      </div>

      {/* Ranking + Matrix */}
      <div className="flex-1 min-h-0 grid grid-cols-1 xl:grid-cols-[1fr_1.3fr] gap-3">
        <RankingTable summary={sum} />
        <PreferenceMatrix summary={sum} />
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-xl shadow-md border border-slate-200 p-4">
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1 text-3xl font-extrabold text-slate-900 truncate">{value}</p>
    </div>
  );
}

function RankingTable({ summary }: { summary: ReturnType<typeof buildSummary> }) {
  if (summary.totalVotes === 0) {
    return (
      <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6 text-slate-400 text-sm">
        Nessun voto salvato.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-md border border-slate-200 flex flex-col min-h-0">
      <div className="px-4 py-3 border-b border-slate-200 flex-shrink-0">
        <h2 className="text-lg font-extrabold uppercase tracking-tight text-slate-800">Ranking</h2>
      </div>
      <div className="overflow-auto flex-1 min-h-0">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-slate-50">
              <Th align="left">Sorgente</Th>
              <Th align="right">Win</Th>
              <Th align="right">Loss</Th>
              <Th align="right">Tie</Th>
              <Th align="right">Win rate</Th>
            </tr>
          </thead>
          <tbody>
            {summary.ranking.map((r) => (
              <tr key={r.id} className="hover:bg-slate-50/50 transition-colors">
                <Td align="left" bold>{r.label}</Td>
                <Td align="right">{r.wins}</Td>
                <Td align="right">{r.losses}</Td>
                <Td align="right">{r.ties}</Td>
                <Td align="right">{Math.round(winRate(r) * 100)}%</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PreferenceMatrix({ summary }: { summary: ReturnType<typeof buildSummary> }) {
  if (summary.totalVotes === 0) {
    return (
      <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6 text-slate-400 text-sm">
        La matrice apparirà dopo i primi confronti.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-md border border-slate-200 flex flex-col min-h-0">
      <div className="px-4 py-3 border-b border-slate-200 flex-shrink-0">
        <h2 className="text-lg font-extrabold uppercase tracking-tight text-slate-800">Preferenze dirette</h2>
      </div>
      <div className="overflow-auto flex-1 min-h-0">
        <table className="min-w-[640px] w-full border-collapse">
          <thead>
            <tr className="bg-slate-50">
              <th className="sticky left-0 z-10 bg-slate-50 text-left text-xs font-bold uppercase tracking-wider text-slate-500 px-3 py-2.5 border-b border-slate-200">
                winner / opponent
              </th>
              {summary.sources.map((s) => (
                <th key={s.id} className="text-center text-xs font-bold uppercase tracking-wider text-slate-500 px-3 py-2.5 border-b border-slate-200 sticky top-0 z-10 bg-slate-50">
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.sources.map((w) => (
              <tr key={w.id} className="hover:bg-slate-50/50 transition-colors">
                <td className="sticky left-0 z-10 bg-white font-bold text-sm text-slate-700 px-3 py-2 border-b border-slate-100">
                  {w.label}
                </td>
                {summary.sources.map((opp) => {
                  const self = w.id === opp.id;
                  const num = self ? 0 : (summary.matrix.get(w.id)?.get(opp.id) || 0);
                  return (
                    <td
                      key={opp.id}
                      className={`text-center text-sm px-3 py-2 border-b border-slate-100 ${
                        self
                          ? 'text-slate-300'
                          : num > 0
                            ? 'font-bold text-accent bg-accent-subtle'
                            : 'text-slate-400'
                      }`}
                    >
                      {self ? '—' : num}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const alignClass = { left: 'text-left', right: 'text-right', center: 'text-center' } as const;

function Th({ children, align = 'left' }: { children: React.ReactNode; align?: 'left' | 'right' | 'center' }) {
  return (
    <th className={`text-xs font-bold uppercase tracking-wider text-slate-500 px-3 py-2.5 border-b border-slate-200 sticky top-0 z-10 bg-slate-50 ${alignClass[align]}`}>
      {children}
    </th>
  );
}

function Td({ children, align = 'left', bold = false }: { children: React.ReactNode; align?: 'left' | 'right' | 'center'; bold?: boolean }) {
  return (
    <td className={`text-sm px-3 py-2 border-b border-slate-100 ${alignClass[align]} ${bold ? 'font-semibold text-slate-800' : 'text-slate-600'}`}>
      {children}
    </td>
  );
}
