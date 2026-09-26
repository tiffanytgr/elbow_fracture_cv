import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

// Writes to the local filesystem → Node.js runtime, never cached.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Absolute path to the JSON Lines study log on the machine running this app.
 * Override with STUDY_LOG_PATH; defaults to `<project>/logs/study-records.jsonl`.
 */
function logFilePath(): string {
  const configured = process.env.STUDY_LOG_PATH?.trim();
  if (configured) {
    return path.isAbsolute(configured)
      ? configured
      : path.join(process.cwd(), configured);
  }
  return path.join(process.cwd(), "logs", "study-records.jsonl");
}

interface StudyPayload {
  reviewer?: string | null;
  mode?: string | null; // "ai" | "control"
  case_id?: string | null;
  ap_path?: string | null;
  lat_path?: string | null;
  input_mode?: string | null;
  pre_grade?: string | null;
  pre_confidence?: number | null;
  post_grade?: string | null;
  post_confidence?: number | null;
  ai_gartland_grade?: string | null;
  ai_cnn_grade?: string | null;
  ai_geometric_grade?: string | null;
  ai_confidence?: number | null;
  notes?: string | null;
  decision_time_seconds?: number | null;
  decision_started_at?: string | null;
  grade_submitted_at?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed_seconds?: number | null;
}

function secondsToHms(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
}

type Entry = Record<string, unknown>;

async function readEntries(filePath: string): Promise<Entry[]> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return raw
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line) as Entry;
        } catch {
          return null;
        }
      })
      .filter((e): e is Entry => e !== null);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw err;
  }
}

function csvField(value: unknown): string {
  if (value === null || value === undefined) return "";
  const s = String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function toCsv(rows: (string | number | null | undefined)[][]): string {
  return rows.map((row) => row.map(csvField).join(",")).join("\r\n") + "\r\n";
}

const RAW_COLUMNS = [
  "logged_at",
  "started_at",
  "ended_at",
  "reviewer",
  "mode",
  "case_id",
  "ap_path",
  "lat_path",
  "input_mode",
  "pre_grade",
  "pre_confidence",
  "post_grade",
  "post_confidence",
  "ai_gartland_grade",
  "ai_cnn_grade",
  "ai_geometric_grade",
  "ai_confidence",
  "decision_started_at",
  "grade_submitted_at",
  "decision_time_seconds",
  "elapsed_seconds",
  "elapsed_hms",
  "notes",
] as const;

function rawCsv(entries: Entry[]): string {
  const rows: (string | number | null | undefined)[][] = [
    [...RAW_COLUMNS],
    ...entries.map((e) =>
      RAW_COLUMNS.map((c) =>
        // Records written before the rename stored the Gartland grade as ai_grade.
        c === "ai_gartland_grade" && e[c] === undefined
          ? (e.ai_grade as string | null)
          : (e[c] as string | number | null),
      ),
    ),
  ];
  return toCsv(rows);
}

interface GroupStats {
  group: string;
  count: number;
  mean_seconds: number;
  median_seconds: number;
  min_seconds: number;
  max_seconds: number;
  total_seconds: number;
  mean_decision_seconds: number | null;
  median_decision_seconds: number | null;
}

function median(sorted: number[]): number {
  const n = sorted.length;
  if (n === 0) return 0;
  return n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

function statsFor(
  group: string,
  seconds: number[],
  decisionSeconds: number[],
): GroupStats {
  const sorted = [...seconds].sort((a, b) => a - b);
  const n = sorted.length;
  const total = sorted.reduce((a, b) => a + b, 0);
  const decisions = [...decisionSeconds].sort((a, b) => a - b);
  const nd = decisions.length;
  const round1 = (x: number) => Math.round(x * 10) / 10;
  return {
    group,
    count: n,
    mean_seconds: n ? round1(total / n) : 0,
    median_seconds: round1(median(sorted)),
    min_seconds: n ? round1(sorted[0]) : 0,
    max_seconds: n ? round1(sorted[n - 1]) : 0,
    total_seconds: round1(total),
    mean_decision_seconds: nd
      ? round1(decisions.reduce((a, b) => a + b, 0) / nd)
      : null,
    median_decision_seconds: nd ? round1(median(decisions)) : null,
  };
}

function summarize(entries: Entry[], groupBy: string): GroupStats[] {
  const secondsOf = (e: Entry) =>
    typeof e.elapsed_seconds === "number" ? e.elapsed_seconds : NaN;
  const timed = entries.filter((e) => isFinite(secondsOf(e)));

  const decisionsOf = (list: Entry[]) =>
    list
      .map((e) => e.decision_time_seconds)
      .filter((v): v is number => typeof v === "number" && isFinite(v));

  const overall = statsFor("ALL", timed.map(secondsOf), decisionsOf(timed));

  const buckets = new Map<string, Entry[]>();
  for (const e of timed) {
    const key = (e[groupBy] as string | null) ?? "(none)";
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(e);
  }
  const groups = Array.from(buckets.entries())
    .map(([key, list]) => statsFor(key, list.map(secondsOf), decisionsOf(list)))
    .sort((a, b) => a.group.localeCompare(b.group));

  return [overall, ...groups];
}

const SUMMARY_COLUMNS: (keyof GroupStats)[] = [
  "group",
  "count",
  "mean_seconds",
  "median_seconds",
  "min_seconds",
  "max_seconds",
  "total_seconds",
  "mean_decision_seconds",
  "median_decision_seconds",
];

function summaryCsv(stats: GroupStats[], groupLabel: string): string {
  const header = ["group_" + groupLabel, ...SUMMARY_COLUMNS.slice(1)];
  const rows: (string | number | null)[][] = [
    header,
    ...stats.map((s) => SUMMARY_COLUMNS.map((c) => s[c])),
  ];
  return toCsv(rows);
}

function csvResponse(body: string, filename: string): NextResponse {
  return new NextResponse(body, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${filename}"`,
    },
  });
}

/**
 * POST /api/study-log
 * Appends one study record as a JSON line to the local log file.
 * Requires mode and elapsed_seconds; a pre-AI grade is required for every arm.
 */
export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as StudyPayload;

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
    if (body.mode !== "ai" && body.mode !== "control") {
      return NextResponse.json(
        { error: "mode is required and must be 'ai' or 'control'" },
        { status: 400 },
      );
    }
    if (!body.pre_grade) {
      return NextResponse.json(
        { error: "pre_grade is required" },
        { status: 400 },
      );
    }

    const num = (v: number | null | undefined) =>
      typeof v === "number" && isFinite(v) ? v : null;
    const decisionSeconds = num(body.decision_time_seconds);

    const entry = {
      reviewer: body.reviewer?.trim() || null,
      mode: body.mode,
      case_id: body.case_id ?? null,
      ap_path: body.ap_path ?? null,
      lat_path: body.lat_path ?? null,
      input_mode: body.input_mode ?? null,
      pre_grade: body.pre_grade,
      pre_confidence: num(body.pre_confidence),
      // Post-AI answer only applies to the AI arm.
      post_grade: body.mode === "ai" ? body.post_grade ?? null : null,
      post_confidence: body.mode === "ai" ? num(body.post_confidence) : null,
      ai_gartland_grade:
        body.mode === "ai" ? body.ai_gartland_grade ?? null : null,
      ai_cnn_grade: body.mode === "ai" ? body.ai_cnn_grade ?? null : null,
      ai_geometric_grade:
        body.mode === "ai" ? body.ai_geometric_grade ?? null : null,
      ai_confidence: body.mode === "ai" ? num(body.ai_confidence) : null,
      notes: body.notes?.trim() || null,
      decision_started_at: body.decision_started_at ?? null,
      grade_submitted_at: body.grade_submitted_at ?? null,
      decision_time_seconds:
        decisionSeconds === null
          ? null
          : Math.round(Math.max(0, decisionSeconds) * 1000) / 1000,
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
      { error: "Failed to write study log", detail: message },
      { status: 500 },
    );
  }
}

/**
 * GET /api/study-log
 *   ?limit=10                 → recent records as JSON (default)
 *   ?format=csv               → all records as a CSV download
 *   ?summary=1[&group_by=…]   → grouped timing stats as JSON
 *   ?summary=1&format=csv     → grouped stats as a CSV download
 * group_by defaults to mode (study arm); reviewer / pre_grade / input_mode also work.
 */
export async function GET(req: NextRequest) {
  const filePath = logFilePath();
  const params = req.nextUrl.searchParams;
  const format = params.get("format");
  const wantSummary =
    params.get("summary") === "1" || params.get("summary") === "true";
  const groupBy = params.get("group_by") || "mode";

  try {
    const entries = await readEntries(filePath);

    if (wantSummary) {
      const stats = summarize(entries, groupBy);
      if (format === "csv") {
        return csvResponse(summaryCsv(stats, groupBy), "study-summary.csv");
      }
      return NextResponse.json({
        log_path: filePath,
        count: entries.length,
        group_by: groupBy,
        summary: stats,
      });
    }

    if (format === "csv") {
      return csvResponse(rawCsv(entries), "study-records.csv");
    }

    const limit = Math.min(
      Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1),
      200,
    );
    return NextResponse.json({
      log_path: filePath,
      count: entries.length,
      entries: entries.slice(-limit).reverse(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Failed to read study log", detail: message },
      { status: 500 },
    );
  }
}
