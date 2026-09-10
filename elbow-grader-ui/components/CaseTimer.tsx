"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Check,
  History,
  Pause,
  Play,
  RotateCcw,
  Save,
  Timer as TimerIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";

const REVIEWER_STORAGE_KEY = "elbow-grader-reviewer";

interface RecentEntry {
  case_id: string | null;
  reviewer: string | null;
  elapsed_hms: string;
  final_grade: string | null;
  logged_at: string;
}

interface CaseTimerProps {
  /** Stable identity of the loaded case. Changing it resets & restarts the timer. */
  caseKey: string | null;
  /** Human-readable id/label for the current case (e.g. demo id or filenames). */
  caseId: string | null;
  /** How the case was loaded, stored alongside the timing. */
  inputMode: string;
  /** Optional analysis outcome, stored alongside the timing when available. */
  finalGrade?: string | null;
  confidence?: number | null;
}

function formatHms(totalMs: number): string {
  const s = Math.max(0, Math.floor(totalMs / 1000));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
}

export function CaseTimer({
  caseKey,
  caseId,
  inputMode,
  finalGrade = null,
  confidence = null,
}: CaseTimerProps) {
  const [running, setRunning] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [reviewer, setReviewer] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);

  // Accumulated milliseconds from previous run segments, plus the timestamp
  // the current segment started (null while paused/stopped).
  const accumulatedRef = useRef(0);
  const segmentStartRef = useRef<number | null>(null);
  const startedAtRef = useRef<string | null>(null);

  // Restore the reviewer name from a previous session.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(REVIEWER_STORAGE_KEY);
      if (stored) setReviewer(stored);
    } catch {
      /* localStorage may be unavailable */
    }
  }, []);

  const loadRecent = useCallback(async () => {
    try {
      const res = await fetch("/api/timer-log?limit=5", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      setRecent(Array.isArray(data.entries) ? data.entries : []);
      setLogPath(data.log_path ?? null);
    } catch {
      /* ignore — history is best-effort */
    }
  }, []);

  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  // Auto reset & start whenever a new case is loaded.
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
    setSavedAt(null);
    setSaveError(null);
    setRunning(true);
  }, [caseKey]);

  // Tick while running.
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

  function currentElapsedSeconds(): number {
    const base = accumulatedRef.current;
    const seg =
      running && segmentStartRef.current !== null
        ? Date.now() - segmentStartRef.current
        : 0;
    return (base + seg) / 1000;
  }

  async function saveTiming() {
    if (!caseKey) return;
    setSaving(true);
    setSaveError(null);

    // Freeze the timer at the moment of saving.
    if (running && segmentStartRef.current !== null) {
      accumulatedRef.current += Date.now() - segmentStartRef.current;
      segmentStartRef.current = null;
    }
    const elapsedSeconds = accumulatedRef.current / 1000;
    setElapsedMs(accumulatedRef.current);
    setRunning(false);

    try {
      const trimmedReviewer = reviewer.trim();
      try {
        if (trimmedReviewer) {
          window.localStorage.setItem(REVIEWER_STORAGE_KEY, trimmedReviewer);
        }
      } catch {
        /* ignore */
      }

      const res = await fetch("/api/timer-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: caseId,
          reviewer: trimmedReviewer || null,
          notes: notes.trim() || null,
          input_mode: inputMode,
          final_grade: finalGrade,
          confidence,
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
      setNotes("");
      setLogPath(data.log_path ?? logPath);
      await loadRecent();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to save timing");
    } finally {
      setSaving(false);
    }
  }

  const disabled = !caseKey;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <TimerIcon className="h-5 w-5 text-blue-600" />
          <h2 className="text-base font-semibold">Case Review Timer</h2>
        </div>
        <div
          className="font-mono text-3xl font-bold tabular-nums tracking-tight text-slate-900"
          aria-live="off"
        >
          {formatHms(elapsedMs)}
        </div>
      </div>

      <p className="mt-1 text-sm text-muted-foreground">
        {disabled
          ? "Load a case to start timing your review."
          : running
            ? `Timing “${caseId ?? "current case"}” — pause or save when the review is done.`
            : `Paused at ${formatHms(elapsedMs)}. Resume or save this review.`}
      </p>

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
          Reset
        </Button>
        <Button
          size="sm"
          onClick={saveTiming}
          disabled={disabled || saving}
          className="gap-1.5 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
        >
          <Save className="h-4 w-4" />
          {saving ? "Saving…" : "Save time to log"}
        </Button>
        {savedAt && !saveError && (
          <span className="inline-flex items-center gap-1 text-sm font-medium text-green-700">
            <Check className="h-4 w-4" />
            Saved at {savedAt}
          </span>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="text-sm">
          <span className="mb-1 block font-medium text-slate-700">
            Reviewer
          </span>
          <input
            type="text"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            placeholder="Your name or initials"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block font-medium text-slate-700">
            Notes (optional)
          </span>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Anything worth recording for this case"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </label>
      </div>

      {saveError && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          <strong>Could not save:</strong> {saveError}
        </p>
      )}

      {recent.length > 0 && (
        <div className="mt-5 border-t border-slate-100 pt-4">
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
                  {entry.final_grade ? ` · Grade ${entry.final_grade}` : ""}
                  {entry.reviewer ? ` · ${entry.reviewer}` : ""}
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
