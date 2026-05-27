import { readdir, readFile, stat } from "node:fs/promises";
import { join, basename } from "node:path";

const MINI_IMAGENET_SIMPLE_LABELS = {
  0: "house finch", 1: "robin", 2: "triceratops", 3: "green mamba",
  4: "harvester", 5: "hen", 6: "goldfinch", 7: "bald eagle",
  8: "vulture", 9: "great grey owl", 10: "European fire salamander",
  11: "smooth newt", 12: "newt", 13: "spotted salamander",
  14: "axolotl", 15: "bullfrog", 16: "tree frog", 17: "tailed frog",
  18: "loggerhead", 19: "leatherback turtle", 20: "mud turtle",
  21: "terrapin", 22: "box turtle", 23: "banded gecko",
  24: "common iguana", 25: "American chameleon", 26: "whiptail",
  27: "lion", 28: "fox squirrel", 29: "marmot", 30: "beaver",
  31: "guinea pig", 32: "sorrel", 33: "zebra", 34: "rock beauty",
  35: "aircraft carrier", 36: "ashcan", 37: "barrel", 38: "basketball",
  39: "bathtub", 40: "beacon", 41: "beer bottle", 42: "bikini",
  43: "binoculars", 44: "birdhouse", 45: "bow tie", 46: "brass",
  47: "broom", 48: "bucket", 49: "candle", 50: "cannon",
  51: "canoe", 52: "electric guitar", 53: "file", 54: "fireboat",
  55: "flagpole", 56: "flute", 57: "folding chair", 58: "frying pan",
  59: "fur coat", 60: "goblet", 61: "go-kart", 62: "gown",
  63: "guillotine", 64: "hamper", 65: "harmonica", 66: "honeycomb",
  67: "iron", 68: "jack-o-lantern", 69: "jean", 70: "jeep",
  71: "knee pad", 72: "lamp", 73: "lawn mower", 74: "lifeboat",
  75: "limousine", 76: "mashed potato", 77: "mattress", 78: "microphone",
  79: "microwave", 80: "motor scooter", 81: "tank", 82: "nail",
  83: "parking meter", 84: "pillow", 85: "ping-pong ball",
  86: "pirate", 87: "pitcher", 88: "potpie", 89: "racket",
  90: "reel", 91: "revolver", 92: "consomme", 93: "saltshaker",
  94: "hot dog", 95: "snorkel", 96: "sock", 97: "sombrero",
  98: "space heater", 99: "ear of corn"
};

export class ArenaStore {
  constructor({ reports, seed }) {
    if (!reports.length) throw new Error("Nessun report JSON trovato per l'arena.");
    this._reportPaths = reports;
    this.rng = createSeededRng(seed);
    this.sources = new Map();
    this.answers = new Map();
    this.records = new Map();
    this.prompts = new Map();
  }

  async init() {
    await this._loadReports();
    return this;
  }

  get sourceIds() {
    return [...this.sources.keys()];
  }

  summary() {
    const sampleCount = {};
    for (const id of this.sourceIds) {
      sampleCount[id] = this.answers.get(id).size;
    }
    const commonAll = this._commonKeys(this.sourceIds);
    return {
      sources: this.sourceIds.map((id) => {
        const s = this.sources.get(id);
        return {
          id: s.sourceId,
          label: s.label,
          kind: s.kind,
          report_path: s.reportPath,
          output_key: s.outputKey,
          num_samples: sampleCount[id],
        };
      }),
      num_sources: this.sources.size,
      num_common_samples: commonAll.length,
      default_pair: this._defaultPair(),
    };
  }

  randomSample({ sourceA, sourceB, randomSources }) {
    if (randomSources) {
      [sourceA, sourceB] = this._randomSourcePair();
    } else {
      sourceA = sourceA || "teacher";
      sourceB = sourceB || this._firstNonTeacher();
    }
    if (sourceA === sourceB) throw new Error("Seleziona due sorgenti diverse.");
    if (!this.sources.has(sourceA)) throw new Error(`Sorgente non trovata: ${sourceA}`);
    if (!this.sources.has(sourceB)) throw new Error(`Sorgente non trovata: ${sourceB}`);

    const commonKeys = this._commonKeys([sourceA, sourceB]);
    if (!commonKeys.length) throw new Error("Nessuna immagine condivisa con descrizioni per la coppia selezionata.");

    const key = commonKeys[Math.floor(this.rng() * commonKeys.length)];
    const pair = [sourceA, sourceB];
    if (this.rng() > 0.5) pair.reverse();
    const [leftId, rightId] = pair;

    const record = this.records.get(key);
    const label = record.label ?? -1;
    const imageKey = `${record.split}:${record.sourceIndex ?? record.datasetIndex}`;
    return {
      sample: {
        split: record.split,
        dataset_index: record.datasetIndex,
        source_index: record.sourceIndex ?? record.datasetIndex,
        label,
        label_name: MINI_IMAGENET_SIMPLE_LABELS[label] ?? `class ${label}`,
        prompt: this.prompts.get(key) || "",
        image_url: `/api/image/${encodeURIComponent(imageKey)}`,
      },
      comparison: {
        left: { side: "left", blind_name: "Risposta A", text: this.answers.get(leftId).get(key) },
        right: { side: "right", blind_name: "Risposta B", text: this.answers.get(rightId).get(key) },
      },
      reveal: {
        left: this._sourcePayload(leftId),
        right: this._sourcePayload(rightId),
      },
      pool: {
        source_a: this._sourcePayload(sourceA),
        source_b: this._sourcePayload(sourceB),
        random_sources: randomSources,
        eligible_samples: commonKeys.length,
      },
    };
  }

  async _loadReports() {
    let teacherLoaded = false;
    for (const reportPath of this._reportPaths) {
      const raw = await readFile(reportPath, "utf-8");
      const report = JSON.parse(raw);
      const results = report.results;
      if (!Array.isArray(results)) throw new Error(`Report senza lista results: ${reportPath}`);

      if (!teacherLoaded) {
        const teacherSource = {
          sourceId: "teacher",
          label: "Teacher Qwen",
          kind: "teacher",
          reportPath,
          outputKey: "teacher",
        };
        this.sources.set("teacher", teacherSource);
        this.answers.set("teacher", new Map());
        this._loadAnswersForSource("teacher", results, String(report.prompt || ""));
        teacherLoaded = true;
      }

      const sourceId = sourceIdFromReport(reportPath);
      const label = labelFromReport(reportPath);
      const source = {
        sourceId,
        label,
        kind: "student",
        reportPath,
        outputKey: "student",
      };
      if (this.sources.has(sourceId)) throw new Error(`Sorgente duplicata: ${sourceId}`);
      this.sources.set(sourceId, source);
      this.answers.set(sourceId, new Map());
      this._loadAnswersForSource(sourceId, results, String(report.prompt || ""));
    }
  }

  _loadAnswersForSource(sourceId, results, reportPrompt) {
    const outputKey = this.sources.get(sourceId).outputKey;
    for (const item of results) {
      if (typeof item !== "object" || item === null) continue;
      const split = String(item.split || "test");
      const datasetIndex = item.dataset_index;
      const text = item[outputKey];
      if (datasetIndex == null || typeof text !== "string" || !text.trim()) continue;

      const key = `${split}:${datasetIndex}`;
      this.answers.get(sourceId).set(key, text.trim());

      if (!this.records.has(key)) {
        this.records.set(key, {
          split,
          datasetIndex: Number(datasetIndex),
          sourceIndex: Number(item.source_index ?? datasetIndex),
          label: Number(item.label ?? -1),
        });
      }
      if (!this.prompts.has(key)) {
        this.prompts.set(key, String(item.prompt || reportPrompt));
      }
    }
  }

  _commonKeys(sourceIds) {
    if (!sourceIds.length) return [];
    const sets = sourceIds.map((id) => this.answers.get(id));
    let common = new Set(sets[0].keys());
    for (let i = 1; i < sets.length; i++) {
      const next = sets[i];
      for (const k of common) {
        if (!next.has(k)) common.delete(k);
      }
    }
    return [...common].sort();
  }

  _defaultPair() {
    return ["teacher", this._firstNonTeacher()];
  }

  _firstNonTeacher() {
    for (const id of this.sourceIds) {
      if (id !== "teacher") return id;
    }
    throw new Error("Serve almeno una sorgente student oltre al teacher.");
  }

  _randomSourcePair() {
    if (this.sources.size < 2) throw new Error("Servono almeno due sorgenti per la modalità random.");
    const ids = this.sourceIds;
    for (let i = 0; i < 100; i++) {
      const a = ids[Math.floor(this.rng() * ids.length)];
      let b = ids[Math.floor(this.rng() * ids.length)];
      if (a === b) continue;
      if (this._commonKeys([a, b]).length > 0) return [a, b];
    }
    throw new Error("Non ho trovato una coppia con immagini condivise.");
  }

  _sourcePayload(sourceId) {
    const s = this.sources.get(sourceId);
    return { id: s.sourceId, label: s.label, kind: s.kind, report_path: s.reportPath };
  }
}

// --- Helpers ---

function createSeededRng(seed) {
  let s = seed | 0;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

export async function resolveReports(explicitReports, reportsRoot) {
  if (explicitReports.length > 0) {
    const valid = [];
    for (const p of explicitReports) {
      try {
        await stat(p);
        valid.push(p);
      } catch { /* skip missing */ }
    }
    return valid;
  }

  const entries = await readdir(reportsRoot).catch(() => []);
  const candidates = entries
    .filter((f) => f.startsWith("qwen_report_") && f.endsWith(".json"))
    .map((f) => join(reportsRoot, f))
    .sort();

  const selected = new Map();
  for (const filePath of candidates) {
    const [experimentKey, priority] = reportPreferenceKey(filePath);
    const current = selected.get(experimentKey);
    if (!current || priority < current[0]) {
      selected.set(experimentKey, [priority, filePath]);
    }
  }

  return [...selected.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, [, path]]) => path);
}

function reportPreferenceKey(filePath) {
  const stem = basename(filePath, ".json");
  if (stem.endsWith("_judged")) return [stem.replace(/_judged$/, ""), 1];
  if (stem.endsWith("_bertscore")) return [stem.replace(/_bertscore$/, ""), 2];
  return [stem, 0];
}

export function sourceIdFromReport(filePath) {
  let [stem] = reportPreferenceKey(filePath);
  stem = stem.replace(/^qwen_report_/, "");
  return stem.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export function labelFromReport(filePath) {
  let [stem] = reportPreferenceKey(filePath);
  stem = stem.replace(/^qwen_report_/, "");
  const parts = stem.split("_");
  if (parts.length >= 3) {
    const teacher = parts[0];
    const loss = parts.slice(2).join(" ");
    return `${teacher} / ${loss}`;
  }
  return stem.replace(/_/g, " ");
}
