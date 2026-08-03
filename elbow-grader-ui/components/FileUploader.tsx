"use client";

import { useEffect, useId, useRef, useState, DragEvent, ChangeEvent } from "react";
import Image from "next/image";
import { Maximize2, RotateCcw, RotateCw, UploadCloud, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploaderProps {
  label: string;
  hint?: string;
  accept?: string;
  file: File | null;
  disabled?: boolean;
  onFileChange: (file: File | null) => void;
}

export function FileUploader({
  label,
  hint,
  accept = "image/png,image/jpeg",
  file,
  disabled = false,
  onFileChange,
}: FileUploaderProps) {
  const previewTitleId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [rotating, setRotating] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      setExpanded(false);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!expanded) return;

    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpanded(false);
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [expanded]);

  function handleFile(f: File | null) {
    if (disabled) return;
    onFileChange(f);
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    handleFile(e.target.files?.[0] ?? null);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (disabled) return;
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  }

  async function rotate(degrees: 90 | -90) {
    if (!previewUrl || !file || rotating || disabled) return;
    setRotating(true);
    try {
      const img = new window.Image();
      img.src = previewUrl;
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
      });

      const canvas = document.createElement("canvas");
      // For ±90° the output dimensions are swapped
      canvas.width = img.height;
      canvas.height = img.width;

      const ctx = canvas.getContext("2d")!;
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate((degrees * Math.PI) / 180);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);

      canvas.toBlob((blob) => {
        if (!blob) { setRotating(false); return; }
        const mimeType = file.type === "image/png" ? "image/png" : "image/jpeg";
        const rotated = new File([blob], file.name, { type: mimeType });
        onFileChange(rotated);
        setRotating(false);
      }, file.type);
    } catch {
      setRotating(false);
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-sm font-semibold">{label}</p>

      {previewUrl && file ? (
        <div className="space-y-2">
          <div className="relative w-full max-w-[520px]">
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="group relative block w-full overflow-hidden rounded-lg border bg-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
              aria-label={`Expand ${label} X-ray`}
            >
              <Image
                src={previewUrl}
                alt={file.name}
                width={520}
                height={340}
                className="h-[260px] w-full object-contain transition-transform group-hover:scale-[1.02] sm:h-[300px] xl:h-[340px]"
              />
              <span className="absolute bottom-2 right-2 flex h-8 w-8 items-center justify-center rounded-md bg-black/65 text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
                <Maximize2 className="h-4 w-4" />
              </span>
            </button>
            <button
              type="button"
              onClick={() => handleFile(null)}
              disabled={disabled}
              className="absolute top-1 right-1 bg-white/80 hover:bg-white rounded-full w-6 h-6 flex items-center justify-center text-slate-500 hover:text-red-500 text-xs font-bold shadow"
              aria-label="Remove file"
            >
              ✕
            </button>
          </div>

          {/* Rotation controls */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => rotate(-90)}
              disabled={rotating || disabled}
              className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 transition-colors"
              aria-label="Rotate left 90°"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Rotate left
            </button>
            <button
              type="button"
              onClick={() => rotate(90)}
              disabled={rotating || disabled}
              className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 transition-colors"
              aria-label="Rotate right 90°"
            >
              <RotateCw className="w-3.5 h-3.5" />
              Rotate right
            </button>
            {rotating && <span className="text-xs text-muted-foreground">Rotating…</span>}
          </div>

          <p className="text-xs text-muted-foreground">✓ {file.name}</p>
        </div>
      ) : (
        <div
          role="button"
          tabIndex={disabled ? -1 : 0}
          aria-disabled={disabled}
          onClick={() => !disabled && inputRef.current?.click()}
          onKeyDown={(e) => {
            if (!disabled && e.key === "Enter") inputRef.current?.click();
          }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cn(
            "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 cursor-pointer transition-colors text-center",
            disabled && "cursor-not-allowed opacity-60",
            dragging
              ? "border-primary bg-primary/5"
              : "border-slate-300 bg-slate-50 hover:border-primary/60 hover:bg-primary/5",
          )}
        >
          <UploadCloud className="w-8 h-8 text-slate-400" />
          <p className="text-sm text-slate-500">
            Drop image here or <span className="text-primary font-medium">browse</span>
          </p>
          {hint && <p className="text-xs text-slate-400">{hint}</p>}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        className="hidden"
        onChange={onInputChange}
      />

      {expanded && previewUrl && file && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 p-3 backdrop-blur-sm sm:p-6"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setExpanded(false);
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby={previewTitleId}
            className="flex h-[min(92vh,900px)] w-full max-w-7xl flex-col overflow-hidden rounded-xl border border-white/15 bg-slate-950 shadow-2xl"
          >
            <div className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3 sm:px-5">
              <div className="min-w-0">
                <h3
                  id={previewTitleId}
                  className="truncate font-semibold text-white"
                >
                  {label}
                </h3>
                <p className="truncate text-xs text-slate-400">{file.name}</p>
              </div>
              <button
                type="button"
                onClick={() => setExpanded(false)}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-300 transition hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
                aria-label="Close full X-ray preview"
                autoFocus
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="relative min-h-0 flex-1">
              <Image
                src={previewUrl}
                alt={`${label} full X-ray`}
                fill
                priority
                sizes="100vw"
                className="object-contain p-2 sm:p-4"
              />
            </div>

            <div className="border-t border-white/10 px-4 py-2.5 text-center text-xs text-slate-400">
              Press Esc or click outside to close
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
