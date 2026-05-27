import { createContext, useContext, useCallback, useState, type ReactNode } from 'react';
import type { ArenaSource, SampleResponse, Vote, VoteChoice } from './types';

interface AppState {
  sources: ArenaSource[];
  sample: SampleResponse | null;
  view: 'arena' | 'dashboard';
  votes: Vote[];
  statusText: string;
  busy: boolean;
  revealedChoice: VoteChoice | null;
}

interface AppContextType {
  state: AppState;
  setSources: (sources: ArenaSource[], statusText: string) => void;
  setSample: (sample: SampleResponse | null) => void;
  setView: (view: 'arena' | 'dashboard') => void;
  addVote: (vote: Vote) => void;
  loadVotes: (votes: Vote[]) => void;
  setStatus: (statusText: string) => void;
  setBusy: (busy: boolean) => void;
  reveal: (choice: VoteChoice) => void;
  resetReveal: () => void;
}

function initialState(): AppState {
  return {
    sources: [],
    sample: null,
    view: 'arena',
    votes: [],
    statusText: 'Caricamento report...',
    busy: false,
    revealedChoice: null,
  };
}

const Ctx = createContext<AppContextType | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<AppState>(initialState);

  const setSources = useCallback((sources: ArenaSource[], statusText: string) => {
    setS((p) => ({ ...p, sources, statusText }));
  }, []);
  const setSample = useCallback((sample: SampleResponse | null) => {
    setS((p) => ({ ...p, sample, revealedChoice: null }));
  }, []);
  const setView = useCallback((view: 'arena' | 'dashboard') => {
    setS((p) => ({ ...p, view }));
  }, []);
  const addVote = useCallback((vote: Vote) => {
    setS((p) => ({ ...p, votes: [...p.votes, vote] }));
  }, []);
  const loadVotes = useCallback((votes: Vote[]) => {
    setS((p) => ({ ...p, votes }));
  }, []);
  const setStatus = useCallback((statusText: string) => {
    setS((p) => ({ ...p, statusText }));
  }, []);
  const setBusy = useCallback((busy: boolean) => {
    setS((p) => ({ ...p, busy }));
  }, []);
  const reveal = useCallback((choice: VoteChoice) => {
    setS((p) => ({ ...p, revealedChoice: choice }));
  }, []);
  const resetReveal = useCallback(() => {
    setS((p) => ({ ...p, revealedChoice: null }));
  }, []);

  return (
    <Ctx.Provider value={{ state: s, setSources, setSample, setView, addVote, loadVotes, setStatus, setBusy, reveal, resetReveal }}>
      {children}
    </Ctx.Provider>
  );
}

export function useApp(): AppContextType {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
