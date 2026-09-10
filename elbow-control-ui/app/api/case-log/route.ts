import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

// Writes to the local filesystem → Node.js runtime, never cached.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Absolute path to the JSON Lines log on the machine running this app.
 * Override with CONTROL_LOG_PATH; defaults to `<project>/logs/assessments.jsonl`.
 */
function logFilePath(): string {
  const configured = process.env.CONTROL_LOG_PATH?.trim();
  if (configured) {
    return path.isAbsolute(configured)
      ? configured
      : path.join(process.cwd(), configured);
  }
  return path.join(process.cwd(), "logs", "assessments.jsonl");
}

interface AssessmentPayload {
  case_id?: string | null;
  clinician?: string | null;
  grade?: string | null;
  confidence?: number | null;
  notes?: string | null;
  input_mode?: string | null;
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
  "case_id",
  "clinician",
  "grade",
  "confidence",
  "input_mode",
  "elapsed_seconds",
  "elapsed_hms",
  "notes",
] as const;

function rawCsv(entries: Entry[]): string {
  const rows: (string | number | null | undefined)[][] = [
    [...RAW_COLUMNS],
    ...entries.map((e) =>
      RAW_COLUMNS.map((c) => e[c] as string | number | null),
    ),
  ];
  return toCsv(rows);
}

interface GroupStats {
  group: string;
  count: number;
  mean_confidence: number;
  total_seconds: number;
  mean_seconds: number;
  median_seconds: number;
  min_seconds: number;
  max_seconds: number;
}

function statsFor(
  group: string,
  seconds: number[],
  confidences: number[],
): GroupStats {
  const sorted = [...seconds].sort((a, b) => a - b);
  const n = sorted.length;
  const total = sorted.reduce((a, b) => a + b, 0);
  const median =
    n === 0
      ? 0
      : n % 2
        ? sorted[(n - 1) / 2]
        : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const round1 = (x: number) => Math.round(x * 10) / 10;
  const round2 = (x: number) => Math.round(x * 100) / 100;
  const confN = confidences.length;
  const confMean = confN
    ? confidences.reduce((a, b) => a + b, 0) / confN
    : 0;
  return {
    group,
    count: n,
    mean_confidence: round2(confMean),
    total_seconds: round1(total),
    mean_seconds: n ? round1(total / n) : 0,
    median_seconds: round1(median),
    min_seconds: n ? round1(sorted[0]) : 0,
    max_seconds: n ? round1(sorted[n - 1]) : 0,
  };
}

function summarize(entries: Entry[], groupBy: string): GroupStats[] {
  const secondsOf = (e: Entry) =>
    typeof e.elapsed_seconds === "number" ? e.elapsed_seconds : NaN;
  const confOf = (e: Entry) =>
    typeof e.confidence === "number" ? e.confidence : NaN;
  const timed = entries.filter((e) => isFinite(secondsOf(e)));

  const overall = statsFor(
    "ALL",
    timed.map(secondsOf),
    timed.map(confOf).filter(isFinite),
  );

  const buckets = new Map<string, { secs: number[]; confs: number[] }>();
  for (const e of timed) {
    const key = (e[groupBy] as string | null) ?? "(none)";
    if (!buckets.has(key)) buckets.set(key, { secs: [], confs: [] });
    const b = buckets.get(key)!;
    b.secs.push(secondsOf(e));
    if (isFinite(confOf(e))) b.confs.push(confOf(e));
  }
  const groups = Array.from(buckets.entries())
    .map(([key, b]) => statsFor(key, b.secs, b.confs))
    .sort((a, b) => a.group.localeCompare(b.group));

  return [overall, ...groups];
}

const SUMMARY_COLUMNS: (keyof GroupStats)[] = [
  "group",
  "count",
  "mean_confidence",
  "total_seconds",
  "mean_seconds",
  "median_seconds",
  "min_seconds",
  "max_seconds",
];

function summaryCsv(stats: GroupStats[], groupLabel: string): string {
  const header = ["group_" + groupLabel, ...SUMMARY_COLUMNS.slice(1)];
  const rows: (string | number)[][] = [
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
 * POST /api/case-log
 * Appends one clinician assessment as a JSON line to the local log file.
 */
export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as AssessmentPayload;

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
    if (!body.grade) {
      return NextResponse.json(
        { error: "grade is required" },
        { status: 400 },
      );
    }

    const entry = {
      case_id: body.case_id ?? null,
      clinician: body.clinician?.trim() || null,
      grade: body.grade,
      confidence:
        typeof body.confidence === "number" ? body.confidence : null,
      notes: body.notes?.trim() || null,
      input_mode: body.input_mode ?? null,
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
      { error: "Failed to write assessment log", detail: message },
      { status: 500 },
    );
  }
}

/**
 * GET /api/case-log
 *   ?limit=N                  → recent records as JSON (default)
 *   ?format=csv               → all records as a CSV download
 *   ?summary=1[&group_by=…]   → grouped stats as JSON
 *   ?summary=1&format=csv     → grouped stats as a CSV download
 * group_by defaults to grade; clinician / input_mode are also useful.
 */
export async function GET(req: NextRequest) {
  const filePath = logFilePath();
  const params = req.nextUrl.searchParams;
  const format = params.get("format");
  const wantSummary =
    params.get("summary") === "1" || params.get("summary") === "true";
  const groupBy = params.get("group_by") || "grade";

  try {
    const entries = await readEntries(filePath);

    if (wantSummary) {
      const stats = summarize(entries, groupBy);
      if (format === "csv") {
        return csvResponse(
          summaryCsv(stats, groupBy),
          "assessments-summary.csv",
        );
      }
      return NextResponse.json({
        log_path: filePath,
        count: entries.length,
        group_by: groupBy,
        summary: stats,
      });
    }

    if (format === "csv") {
      return csvResponse(rawCsv(entries), "assessments.csv");
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
      { error: "Failed to read assessment log", detail: message },
      { status: 500 },
    );
  }
}
