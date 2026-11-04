import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import {
  analyzeText,
  type AnalysisResult,
  type TokenRow,
  type Correction,
  type Highlight,
  type StyleSuggestion,
} from "@/lib/engine/nlpEngine";

export const runtime = "nodejs";
export const maxDuration = 60;

export type { TokenRow, Correction, Highlight, StyleSuggestion, AnalysisResult };

const ENGINE_DIR = path.join(process.cwd(), "pyengine");

/**
 * Pick a Python interpreter that has NLTK/TextBlob installed.
 * The dev server's PATH may differ from the shell's, so prefer the
 * project virtualenv explicitly and fall back to system python3.
 */
function resolvePython(): string {
  const candidates = [
    "/home/z/.venv/bin/python3",
    path.join(process.env.HOME ?? "/home/z", ".venv/bin/python3"),
    "python3",
    "python",
  ];
  for (const c of candidates) {
    try {
      if (c.includes("/") && fs.existsSync(c)) return c;
    } catch {
      /* ignore and try next */
    }
  }
  return "python3";
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const text = (body?.text ?? "").toString();

    if (!text.trim()) {
      return NextResponse.json(
        { error: "Please enter some text to analyze." },
        { status: 400 }
      );
    }
    if (text.length > 5000) {
      return NextResponse.json(
        { error: "Text is too long (max 5000 characters)." },
        { status: 400 }
      );
    }

    return await executeEngine(text);
  } catch {
    return NextResponse.json(
      { error: "Invalid request body." },
      { status: 400 }
    );
  }
}

/**
 * Executes the NLP pipeline:
 * Tries the Python NLP engine first (preserving full NLTK/TextBlob behavior if installed).
 * If Python / NLTK is not available (e.g. on serverless Vercel deployment),
 * seamlessly falls back to the native high-performance TypeScript engine.
 */
async function executeEngine(text: string): Promise<NextResponse> {
  const engineScript = path.join(ENGINE_DIR, "engine.py");

  // If engine.py doesn't exist or we're on a pure cloud runtime without python
  if (!fs.existsSync(engineScript)) {
    const result = analyzeText(text);
    return NextResponse.json(result);
  }

  return new Promise((resolve) => {
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawn(resolvePython(), ["engine.py"], {
        cwd: ENGINE_DIR,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          PYTHONDONTWRITEBYTECODE: "1",
          NLTK_DATA: path.join(ENGINE_DIR, "nltk_data"),
        },
      });
    } catch {
      // Spawn failed (e.g. no python3 binary) -> seamless fallback
      const result = analyzeText(text);
      return resolve(NextResponse.json(result));
    }

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        try {
          proc.kill("SIGKILL");
        } catch {
          /* ignore */
        }
        // Fallback to TS engine if Python process hangs
        const result = analyzeText(text);
        resolve(NextResponse.json(result));
      }
    }, 10_000);

    proc.stdout.on("data", (d) => (stdout += d.toString()));
    proc.stderr.on("data", (d) => (stderr += d.toString()));

    proc.on("error", () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      // Fallback on spawn error (e.g. ENOENT)
      const result = analyzeText(text);
      resolve(NextResponse.json(result));
    });

    proc.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);

      try {
        const parsed = JSON.parse(stdout);
        if (parsed?.error) {
          // If Python engine reported missing nltk or packages, fallback
          const result = analyzeText(text);
          resolve(NextResponse.json(result));
        } else if (code === 0) {
          resolve(NextResponse.json(parsed));
        } else {
          // Non-zero exit -> fallback to native TS engine
          const result = analyzeText(text);
          resolve(NextResponse.json(result));
        }
      } catch {
        // Unparseable output / error -> fallback to native TS engine
        const result = analyzeText(text);
        resolve(NextResponse.json(result));
      }
    });

    try {
      proc.stdin.write(JSON.stringify({ text }));
      proc.stdin.end();
    } catch {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        const result = analyzeText(text);
        resolve(NextResponse.json(result));
      }
    }
  });
}
