import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { statSync } from "node:fs";
import { join, resolve } from "node:path";

const inflight = new Map();

function findPython() {
  const venvPy = resolve(import.meta.dirname, "..", "..", "..", ".venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  try { statSync(venvPy); return venvPy; } catch { /* no venv */ }
  return "python";
}

/**
 * Get a Mini-ImageNet image as PNG bytes by split and source_index.
 * Caches results on disk for fast repeat access.
 */
export async function getImage(split, sourceIndex, cacheDir) {
  const cacheKey = `${split}_${sourceIndex}.png`;
  const cachePath = join(cacheDir, cacheKey);

  if (inflight.has(cachePath)) return inflight.get(cachePath);

  const promise = (async () => {
    try {
      return await readFile(cachePath);
    } catch {
      await mkdir(cacheDir, { recursive: true });
      const png = await spawnPython(split, sourceIndex);
      await writeFile(cachePath, png);
      return png;
    }
  })();

  inflight.set(cachePath, promise);
  try {
    return await promise;
  } finally {
    inflight.delete(cachePath);
  }
}

function spawnPython(split, sourceIndex) {
  return new Promise((resolve, reject) => {
    const script = new URL("./serve_image.py", import.meta.url);
    let scriptPath = script.pathname;
    if (process.platform === "win32") scriptPath = scriptPath.replace(/^\//, "");

    const python = findPython();
    const proc = spawn(python, [scriptPath, split, String(sourceIndex)], { stdio: ["ignore", "pipe", "pipe"] });
    const stdout = [];
    const stderr = [];
    proc.stdout.on("data", (chunk) => stdout.push(chunk));
    proc.stderr.on("data", (chunk) => stderr.push(chunk));

    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) {
        resolve(Buffer.concat(stdout));
      } else {
        reject(new Error(`serve_image.py exited ${code}: ${Buffer.concat(stderr).toString()}`));
      }
    });
  });
}
