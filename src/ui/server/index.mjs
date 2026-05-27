import express from "express";
import { resolve } from "node:path";
import { ArenaStore, resolveReports } from "./store.mjs";
import { getImage } from "./image-converter.mjs";

const DEFAULT_REPORTS_ROOT = "runs/scored_reports";
const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8765;
const DEFAULT_SEED = 42;

async function main() {
  const args = parseArgs();

  const reports = await resolveReports(args.reports, args.reportsRoot);
  if (!reports.length) {
    console.error(
      `Nessun report JSON caricabile trovato.\n` +
      `Directory richiesta: ${args.reportsRoot}\n` +
      `Sono supportati report grezzi, _judged.json e _bertscore.json.`
    );
    process.exit(1);
  }

  const store = new ArenaStore({
    reports,
    seed: args.seed,
  });
  await store.init();

  const staticDir = args.staticDir;
  const cacheDir = resolve(import.meta.dirname, "..", "..", "..", "runs", ".arena-image-cache");

  const app = express();

  // API routes
  app.get("/api/sources", (_req, res) => {
    res.json(store.summary());
  });

  app.get("/api/sample", (req, res) => {
    try {
      const payload = store.randomSample({
        sourceA: req.query.source_a || null,
        sourceB: req.query.source_b || null,
        randomSources: req.query.mode === "random",
      });
      res.json(payload);
    } catch (err) {
      res.status(400).json({ error: err.constructor.name, message: err.message });
    }
  });

  app.get("/api/image/:token", async (req, res) => {
    try {
      const token = decodeURIComponent(req.params.token);
      const [split, rawIndex] = token.split(":", 2);
      const sourceIndex = Number(rawIndex);
      const png = await getImage(split, sourceIndex, cacheDir);
      res.set("Content-Type", "image/png");
      res.set("Cache-Control", "no-store");
      res.send(png);
    } catch (err) {
      res.status(400).json({ error: err.constructor.name, message: err.message });
    }
  });

  // Static files (SPA)
  app.use(express.static(staticDir));
  app.get("*", (_req, res) => {
    res.sendFile(resolve(staticDir, "index.html"));
  });

  app.listen(args.port, args.host, () => {
    console.log(`Arena UI: http://${args.host}:${args.port}`);
    console.log(`Static dir: ${staticDir}`);
    console.log(`Sorgenti caricate: ${store.sources.size}`);
    console.log(`Report: ${reports.length}`);
  });
}

function parseArgs() {
  // Project root is 2 levels up from src/ui/server/
  const projectRoot = resolve(import.meta.dirname, "..", "..", "..");

  const argv = process.argv.slice(2);
  const get = (flag, fallback) => {
    const i = argv.indexOf(flag);
    return i !== -1 && i + 1 < argv.length ? argv[i + 1] : fallback;
  };
  const getAll = (flag) => {
    const results = [];
    for (let i = 0; i < argv.length; i++) {
      if (argv[i] === flag && i + 1 < argv.length) results.push(argv[i + 1]);
    }
    return results;
  };

  const explicitReports = getAll("--report").map((p) => resolve(p));
  const reportsRoot = resolve(projectRoot, get("--reports-root", DEFAULT_REPORTS_ROOT));
  const staticDir = resolve(get("--static-dir", resolve(import.meta.dirname, "..", "arena-app", "dist")));
  const host = get("--host", DEFAULT_HOST);
  const port = Number(get("--port", String(DEFAULT_PORT)));
  const seed = Number(get("--seed", String(DEFAULT_SEED)));

  return { reports: explicitReports, reportsRoot, staticDir, host, port, seed };
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
