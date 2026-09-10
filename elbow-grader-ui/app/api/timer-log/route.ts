import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

// This route writes to the local filesystem, so it must run on the Node.js
// runtime (not the Edge runtime) and never be statically cached.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Absolute path to the JSON Lines log file on the machine running this app.
 * Override with the TIMER_LOG_PATH env var (absolute path recommended).
 * Defaults to `<project>/logs/case-timings.jsonl`.
 */
function logFilePath(): string {
  const configured = process.env.TIMER_LOG_PATH?.trim();
  if (configured) {
    return path.isAbsolute(configured)
      ? configured
      : path.join(process.cwd(), configured);
  }
  return path.join(process.cwd(), "logs", "case-timings.jsonl");
}

interface TimerLogPayload {
  case_id?: string | null;
  reviewer?: string | null;
  notes?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed_seconds?: number | null;
  input_mode?: string | null;
  final_grade?: string | null;
  confidence?: number | null;
}

function secondsToHms(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
}

/**
 * POST /api/timer-log
 * Appends one case-timing record as a JSON line to the local log file.
 */
export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as TimerLogPayload;

    const elapsedSeconds =
      typeof body.elapsed_seconds === "number" && isFinite(body.elapsed_seconds)
        ? Math.max(0, body.elapsed_seconds)
        : null;

    if (elapsedSeconds === null) {
      return NextResponse.json(
        { error: "elapsed_seconds is required and must be a number" },
        { status: 400 },
      );
    }

    const entry = {
      case_id: body.case_id ?? null,
      reviewer: body.reviewer?.trim() || null,
      notes: body.notes?.trim() || null,
      input_mode: body.input_mode ?? null,
      final_grade: body.final_grade ?? null,
      confidence:
        typeof body.confidence === "number" ? body.confidence : null,
      elapsed_seconds: Math.round(elapsedSeconds * 1000) / 1000,
      elapsed_hms: secondsToHms(elapsedSeconds),
      started_at: body.started_at ?? null,
      ended_at: body.ended_at ?? new Date().toISOString(),
      logged_at: new Date().toISOString(),
    };

    const filePath = logFilePath();
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.appendFile(filePath, JSON.stringify(entry) + "\n", "utf8");

    return NextResponse.json({ ok: true, saved: entry, log_path: filePath });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Failed to write timer log", detail: message },
      { status: 500 },
    );
  }
}

/**
 * GET /api/timer-log?limit=10
 * Returns the most recent timing records (newest first) plus the log path.
 */
export async function GET(req: NextRequest) {
  const filePath = logFilePath();
  const limitParam = req.nextUrl.searchParams.get("limit");
  const limit = Math.min(
    Math.max(parseInt(limitParam ?? "10", 10) || 10, 1),
    200,
  );

  try {
    const raw = await fs.readFile(filePath, "utf8");
    const entries = raw
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter((e): e is Record<string, unknown> => e !== null);

    const recent = entries.slice(-limit).reverse();
    return NextResponse.json({
      log_path: filePath,
      count: entries.length,
      entries: recent,
    });
  } catch (err) {
    // Missing file just means nothing has been logged yet.
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({ log_path: filePath, count: 0, entries: [] });
    }
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Failed to read timer log", detail: message },
      { status: 500 },
    );
  }
}
