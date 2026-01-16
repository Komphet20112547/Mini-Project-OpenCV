import fs from "node:fs";
import path from "node:path";

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyFileSync(src, dest) {
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
}

function main() {
  const projectRoot = process.cwd();
  const srcDir = path.join(projectRoot, "node_modules", "onnxruntime-web", "dist");
  const destDir = path.join(projectRoot, "public", "ort");

  if (!fs.existsSync(srcDir)) {
    console.warn(
      `[copy-ort-wasm] Skip: cannot find ${srcDir}. Did you run npm install?`,
    );
    return;
  }

  ensureDir(destDir);

  const files = fs
    .readdirSync(srcDir)
    .filter((f) => f.endsWith(".wasm") || f.endsWith(".mjs"));

  for (const file of files) {
    copyFileSync(path.join(srcDir, file), path.join(destDir, file));
  }

  console.log(`[copy-ort-wasm] Copied ${files.length} files to ${destDir}`);
}

main();


