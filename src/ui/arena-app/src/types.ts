export interface ArenaSource {
  id: string;
  label: string;
  kind: string;
  report_path: string;
  output_key: string;
  num_samples: number;
}

export interface SampleInfo {
  split: string;
  dataset_index: number;
  source_index: number;
  label: number;
  label_name: string;
  prompt: string;
  image_url: string;
}

export interface ComparisonSide {
  side: 'left' | 'right';
  blind_name: string;
  text: string;
}

export interface RevealSource {
  id: string;
  label: string;
  kind: string;
  report_path: string;
}

export interface SampleResponse {
  sample: SampleInfo;
  comparison: { left: ComparisonSide; right: ComparisonSide };
  reveal: { left: RevealSource; right: RevealSource };
  pool: {
    source_a: RevealSource;
    source_b: RevealSource;
    random_sources: boolean;
    eligible_samples: number;
  };
}

export interface Vote {
  timestamp: string;
  choice: 'left' | 'right' | 'tie';
  sample: SampleInfo;
  reveal: { left: RevealSource; right: RevealSource };
  mode: string;
}

export type VoteChoice = 'left' | 'right' | 'tie';

export interface SourceStats {
  id: string;
  label: string;
  wins: number;
  losses: number;
  ties: number;
  appearances: number;
}

export interface VoteSummary {
  totalVotes: number;
  tieVotes: number;
  sources: SourceStats[];
  ranking: SourceStats[];
  topWinner: { label: string; wins: number } | null;
  matrix: Map<string, Map<string, number>>;
}
