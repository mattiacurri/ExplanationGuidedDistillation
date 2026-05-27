import { useCallback, useEffect } from 'react';
import { useApp } from '../store';
import type { ArenaSource, SourceStats, Vote, VoteChoice, VoteSummary } from '../types';

const STORAGE_KEY = 'compvis-arena-votes';

export function useArena() {
  const { state, loadVotes, addVote, setSources, setSample, setStatus, setBusy, reveal, resetReveal } = useApp();

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const arr: Vote[] = JSON.parse(raw);
        if (Array.isArray(arr)) loadVotes(arr);
      }
    } catch { /* ignore */ }
  }, [loadVotes]);

  const fetchSources = useCallback(async () => {
    setBusy(true);
    try {
      const res = await fetch('/api/sources');
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Errore sorgenti');
      setSources(data.sources, `${data.num_sources} sorgenti, ${data.num_common_samples} immagini comuni`);
    } catch (e: unknown) {
      setStatus(e instanceof Error ? e.message : 'Errore sconosciuto');
    } finally {
      setBusy(false);
    }
  }, [setBusy, setStatus, setSources]);

  const fetchSample = useCallback(async () => {
    setBusy(true);
    resetReveal();
    setSample(null as any); // clear current content immediately

    try {
      const res = await fetch('/api/sample?mode=random');
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Errore sample');

      // Preload image before showing anything
      await new Promise<void>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Errore caricamento immagine'));
        img.src = `${data.sample.image_url}?t=${Date.now()}`;
      });

      setSample(data);
    } catch (e: unknown) {
      setStatus(e instanceof Error ? e.message : 'Errore sconosciuto');
    } finally {
      setBusy(false);
    }
  }, [setBusy, setStatus, setSample, resetReveal]);

  const saveVote = useCallback(
    (choice: VoteChoice) => {
      if (!state.sample) return;
      const vote: Vote = {
        timestamp: new Date().toISOString(),
        choice,
        sample: state.sample.sample,
        reveal: state.sample.reveal,
        mode: 'random',
      };
      const updated = [...state.votes, vote];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      addVote(vote);
      reveal(choice);
    },
    [state.sample, state.votes, addVote, reveal],
  );

  return { fetchSources, fetchSample, saveVote };
}

// --- Vote statistics ---

function ensureStat(stats: Map<string, SourceStats>, id: string, label: string) {
  if (!stats.has(id)) stats.set(id, { id, label, wins: 0, losses: 0, ties: 0, appearances: 0 });
}

export function winRate(item: SourceStats): number {
  const d = item.wins + item.losses;
  return d === 0 ? 0 : item.wins / d;
}

export function buildSummary(sources: ArenaSource[], votes: Vote[]): VoteSummary {
  const stats = new Map<string, SourceStats>();
  const matrix = new Map<string, Map<string, number>>();
  let tieVotes = 0;

  for (const s of sources) ensureStat(stats, s.id, s.label);
  for (const v of votes) {
    const L = v.reveal?.left;
    const R = v.reveal?.right;
    if (!L?.id || !R?.id) continue;
    ensureStat(stats, L.id, L.label);
    ensureStat(stats, R.id, R.label);
    stats.get(L.id)!.appearances++;
    stats.get(R.id)!.appearances++;
    if (v.choice === 'tie') {
      tieVotes++;
      stats.get(L.id)!.ties++;
      stats.get(R.id)!.ties++;
      continue;
    }
    const w = v.choice === 'left' ? L : R;
    const l = v.choice === 'left' ? R : L;
    stats.get(w.id)!.wins++;
    stats.get(l.id)!.losses++;
    if (!matrix.has(w.id)) matrix.set(w.id, new Map());
    const row = matrix.get(w.id)!;
    row.set(l.id, (row.get(l.id) || 0) + 1);
  }

  const sourceList = [...stats.values()].sort((a, b) => a.label.localeCompare(b.label));
  const ranking = [...sourceList].sort(
    (a, b) => b.wins - a.wins || winRate(b) - winRate(a) || a.label.localeCompare(b.label),
  );
  const top = ranking.find((x) => x.wins > 0);

  return {
    totalVotes: votes.length,
    tieVotes,
    sources: sourceList,
    ranking,
    topWinner: top ? { label: top.label, wins: top.wins } : null,
    matrix,
  };
}
