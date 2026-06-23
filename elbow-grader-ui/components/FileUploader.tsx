"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";
import Image from "next/image";
import { UploadCloud, RotateCcw, RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploaderProps {
  label: string;
  hint?: string;
  accept?: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export function FileUploader({
  label,
  hint,
  accept = "image/png,image/jpeg",
  file,
  onFileChange,
}: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [rotating, setRotating] = useState(false);

  function handleFile(f: File | null) {
    onFileChange(f);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    } else {
      setPreviewUrl(null);
    }
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    handleFile(e.target.files?.[0] ?? null);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  }

  async function rotate(degrees: 90 | -90) {
    if (!previewUrl || !file || rotating) return;
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
        const url = URL.createObjectURL(rotated);
        setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return url; });
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
          <div className="relative inline-block">
            <div className="rounded-lg border overflow-hidden">
              <Image
                src={previewUrl}
                alt={file.name}
                width={260}
                height={260}
                className="max-w-[260px] h-auto object-contain"
              />
            </div>
            <button
              type="button"
              onClick={() => handleFile(null)}
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
              disabled={rotating}
              className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 transition-colors"
              aria-label="Rotate left 90°"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Rotate left
            </button>
            <button
              type="button"
              onClick={() => rotate(90)}
              disabled={rotating}
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
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cn(
            "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 cursor-pointer transition-colors text-center",
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
        className="hidden"
        onChange={onInputChange}
      />
    </div>
  );
}
