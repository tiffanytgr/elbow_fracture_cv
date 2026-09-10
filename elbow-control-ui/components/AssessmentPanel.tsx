"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  BarChart3,
  Check,
  Download,
  History,
  Pause,
  Play,
  RotateCcw,
  Save,
  Timer as TimerIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  CONFIDENCE_LABELS,
  CONFIDENCE_LEVELS,
  GARTLAND_GRADES,
  type ConfidenceLevel,
  type GartlandGrade,
} from "@/lib/types";

const CLINICIAN_STORAGE_KEY = "elbow-control-clinician";

interface RecentEntry {
  case_id: string | null;
  clinician: string | null;
  grade: string | null;
  confidence: number | null;
  elapsed_hms: string;
  logged_at: string;
}

interface SummaryRow {
  group: string;
  count: number;
  mean_confidence: number;
  mean_seconds: number;
  median_seconds: number;
  total_seconds: number;
}

interface AssessmentPanelProps {
  caseKey: string | null;
  caseId: string | null;
  inputMode: string;
  /** Called after a successful save, so the page can advance to the next case. */
  onSaved?: () => void;
}

function formatHms(totalMs: number): string {
  const s = Math.max(0, Math.floor(totalMs / 1000));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
}

function formatSeconds(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return mm > 0 ? `${mm}m ${String(ss).padStart(2, "0")}s` : `${ss}s`;
}

export function AssessmentPanel({
  caseKey,
  caseId,
  inputMode,
  onSaved,
}: AssessmentPanelProps) {
  const [running, setRunning] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [clinician, setClinician] = useState("");
  const [grade, setGrade] = useState<GartlandGrade | null>(null);
  const [confidence, setConfidence] = useState<ConfidenceLevel | null>(null);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);
  const [summary, setSummary] = useState<SummaryRow[] | null>(null);
  const [showSummary, setShowSummary] = useState(false);

  const accumulatedRef = useRef(0);
  const segmentStartRef = useRef<number | null>(null);
  const startedAtRef = useRef<string | null>(null);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(CLINICIAN_STORAGE_KEY);
      if (stored) setClinician(stored);
    } catch {
      /* localStorage may be unavailable */
    }
  }, []);

  const loadRecent = useCallback(async () => {
    try {
      const res = await fetch("/api/case-log?limit=5", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      setRecent(Array.isArray(data.entries) ? data.entries : []);
      setLogPath(data.log_path ?? null);
    } catch {
      /* best-effort */
    }
  }, []);

  const loadSummary = useCallback(async () => {
    try {
      const res = await fetch("/api/case-log?summary=1&group_by=grade", {
        cache: "no-store",
      });
      if (!res.ok) return;
      const data = await res.json();
      setSummary(Array.isArray(data.summary) ? data.summary : []);
    } catch {
      /* best-effort */
    }
  }, []);

  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  useEffect(() => {
    if (showSummary) loadSummary();
  }, [showSummary, loadSummary]);

  // Auto reset & start when a new case loads; clear the previous grading.
  useEffect(() => {
    if (!caseKey) {
      setRunning(false);
      setElapsedMs(0);
      accumulatedRef.current = 0;
      segmentStartRef.current = null;
      startedAtRef.current = null;
      return;
    }
    accumulatedRef.current = 0;
    segmentStartRef.current = Date.now();
    startedAtRef.current = new Date().toISOString();
    setElapsedMs(0);
    setGrade(null);
    setConfidence(null);
    setNotes("");
    setSavedAt(null);
    setSaveError(null);
    setRunning(true);
  }, [caseKey]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      const base = accumulatedRef.current;
      const seg =
        segmentStartRef.current !== null
          ? Date.now() - segmentStartRef.current
          : 0;
      setElapsedMs(base + seg);
    }, 250);
    return () => window.clearInterval(id);
  }, [running]);

  function pause() {
    if (!running) return;
    if (segmentStartRef.current !== null) {
      accumulatedRef.current += Date.now() - segmentStartRef.current;
      segmentStartRef.current = null;
    }
    setElapsedMs(accumulatedRef.current);
    setRunning(false);
  }

  function resume() {
    if (running || !caseKey) return;
    segmentStartRef.current = Date.now();
    if (!startedAtRef.current) startedAtRef.current = new Date().toISOString();
    setRunning(true);
  }

  function reset() {
    accumulatedRef.current = 0;
    segmentStartRef.current = running ? Date.now() : null;
    startedAtRef.current = running ? new Date().toISOString() : null;
    setElapsedMs(0);
    setSavedAt(null);
    setSaveError(null);
  }

  async function saveAssessment() {
    if (!caseKey) return;
    if (!grade) {
      setSaveError("Select a Gartland grade before saving.");
      return;
    }
    setSaving(true);
    setSaveError(null);

    if (running && segmentStartRef.current !== null) {
      accumulatedRef.current += Date.now() - segmentStartRef.current;
      segmentStartRef.current = null;
    }
    const elapsedSeconds = accumulatedRef.current / 1000;
    setElapsedMs(accumulatedRef.current);
    setRunning(false);

    try {
      const trimmedClinician = clinician.trim();
      try {
        if (trimmedClinician) {
          window.localStorage.setItem(
            CLINICIAN_STORAGE_KEY,
            trimmedClinician,
          );
        }
      } catch {
        /* ignore */
      }

      const res = await fetch("/api/case-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: caseId,
          clinician: trimmedClinician || null,
          grade,
          confidence,
          notes: notes.trim() || null,
          input_mode: inputMode,
          elapsed_seconds: elapsedSeconds,
          started_at: startedAtRef.current,
          ended_at: new Date().toISOString(),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSaveError(data.detail ?? data.error ?? `Error ${res.status}`);
        return;
      }
      setSavedAt(new Date().toLocaleTimeString());
      setLogPath(data.log_path ?? logPath);
      await loadRecent();
      if (showSummary) await loadSummary();
      onSaved?.();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  const disabled = !caseKey;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Your Assessment</h2>
        <div className="flex items-center gap-2 text-slate-900">
          <TimerIcon className="h-5 w-5 text-blue-600" />
          <span className="font-mono text-2xl font-bold tabular-nums tracking-tight">
            {formatHms(elapsedMs)}
          </span>
        </div>
      </div>

      <p className="mt-1 text-sm text-muted-foreground">
        {disabled
          ? "Load a case to begin your assessment."
          : "Review the X-ray(s), record your grade and confidence, then save."}
      </p>

      {/* Gartland grade */}
      <div className="mt-4">
        <span className="mb-1.5 block text-sm font-medium text-slate-700">
          Gartland grade
        </span>
        <div
          role="group"
          aria-label="Gartland grade"
          className="flex flex-wrap gap-2"
        >
          {GARTLAND_GRADES.map((g) => {
            const selected = grade === g;
            return (
              <button
                key={g}
                type="button"
                disabled={disabled}
                aria-pressed={selected}
                onClick={() => setGrade(g)}
                className={`min-w-[3.5rem] rounded-lg border px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  selected
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                {g}
              </button>
            );
          })}
        </div>
      </div>

      {/* Confidence */}
      <div className="mt-4">
        <span className="mb-1.5 block text-sm font-medium text-slate-700">
          Confidence
        </span>
        <div
          role="group"
          aria-label="Confidence"
          className="flex flex-wrap items-center gap-2"
        >
          {CONFIDENCE_LEVELS.map((level) => {
            const selected = confidence === level;
            return (
              <button
                key={level}
                type="button"
                disabled={disabled}
                aria-pressed={selected}
                title={CONFIDENCE_LABELS[level]}
                onClick={() => setConfidence(level)}
                className={`flex h-10 w-10 items-center justify-center rounded-full border text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  selected
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                {level}
              </button>
            );
          })}
          <span className="ml-1 text-xs text-slate-500">
            {confidence ? CONFIDENCE_LABELS[confidence] : "1 = very unsure · 5 = very confident"}
          </span>
        </div>
      </div>

      {/* Clinician + notes */}
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="text-sm">
          <span className="mb-1 block font-medium text-slate-700">
            Clinician
          </span>
          <input
            type="text"
            value={clinician}
            onChange={(e) => setClinician(e.target.value)}
            placeholder="Your name or initials"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </label>
        <label className="text-sm sm:row-span-2">
          <span className="mb-1 block font-medium text-slate-700">
            Notes / reasoning (optional)
          </span>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="Findings, uncertainty, rationale…"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </label>
      </div>

      {/* Timer + save controls */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {running ? (
          <Button
            variant="outline"
            size="sm"
            onClick={pause}
            disabled={disabled}
            className="gap-1.5"
          >
            <Pause className="h-4 w-4" />
            Pause
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={resume}
            disabled={disabled}
            className="gap-1.5"
          >
            <Play className="h-4 w-4" />
            Resume
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={reset}
          disabled={disabled}
          className="gap-1.5"
        >
          <RotateCcw className="h-4 w-4" />
          Reset timer
        </Button>
        <Button
          size="sm"
          onClick={saveAssessment}
          disabled={disabled || saving}
          className="gap-1.5 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
        >
          <Save className="h-4 w-4" />
          {saving ? "Saving…" : "Save assessment"}
        </Button>
        {savedAt && !saveError && (
          <span className="inline-flex items-center gap-1 text-sm font-medium text-green-700">
            <Check className="h-4 w-4" />
            Saved at {savedAt}
          </span>
        )}
      </div>

      {saveError && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          {saveError}
        </p>
      )}

      {recent.length > 0 && (
        <div className="mt-5 border-t border-slate-100 pt-4">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <a
              href="/api/case-log?format=csv"
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-blue-400 hover:text-blue-700"
            >
              <Download className="h-4 w-4" />
              Download log (CSV)
            </a>
            <a
              href="/api/case-log?summary=1&format=csv&group_by=grade"
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-blue-400 hover:text-blue-700"
            >
              <Download className="h-4 w-4" />
              Summary (CSV)
            </a>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSummary((v) => !v)}
              className="gap-1.5"
            >
              <BarChart3 className="h-4 w-4" />
              {showSummary ? "Hide summary" : "View summary"}
            </Button>
          </div>

          {showSummary && summary && summary.length > 0 && (
            <div className="mb-4 overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-semibold">Grade</th>
                    <th className="px-3 py-2 text-right font-semibold">Cases</th>
                    <th className="px-3 py-2 text-right font-semibold">
                      Mean conf.
                    </th>
                    <th className="px-3 py-2 text-right font-semibold">Mean</th>
                    <th className="px-3 py-2 text-right font-semibold">
                      Median
                    </th>
                    <th className="px-3 py-2 text-right font-semibold">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => {
                    const isAll = row.group === "ALL";
                    return (
                      <tr
                        key={row.group}
                        className={
                          isAll
                            ? "border-t border-slate-200 bg-slate-50 font-semibold"
                            : "border-t border-slate-100"
                        }
                      >
                        <td className="px-3 py-2">
                          {isAll ? "All cases" : row.group}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.count}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.mean_confidence || "—"}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatSeconds(row.mean_seconds)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatSeconds(row.median_seconds)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatSeconds(row.total_seconds)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <History className="h-3.5 w-3.5" />
            Recently logged
          </div>
          <ul className="mt-2 space-y-1.5">
            {recent.map((entry, i) => (
              <li
                key={`${entry.logged_at}-${i}`}
                className="flex flex-wrap items-center justify-between gap-2 text-sm"
              >
                <span className="text-slate-700">
                  <span className="font-mono font-semibold">
                    {entry.elapsed_hms}
                  </span>{" "}
                  · {entry.case_id ?? "—"}
                  {entry.grade ? ` · Grade ${entry.grade}` : ""}
                  {entry.confidence ? ` · conf ${entry.confidence}` : ""}
                  {entry.clinician ? ` · ${entry.clinician}` : ""}
                </span>
                <span className="text-xs text-slate-400">
                  {new Date(entry.logged_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
          {logPath && (
            <p className="mt-3 break-all text-xs text-slate-400">
              Saved on this device: <code>{logPath}</code>
            </p>
          )}
        </div>
      )}
    </section>
  );
}
