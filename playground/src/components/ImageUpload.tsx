import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

async function downscaleIfNeeded(dataUrl: string, maxPixels: number): Promise<string> {
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("img load failed"));
    img.src = dataUrl;
  });
  const total = img.naturalWidth * img.naturalHeight;
  if (!total || total <= maxPixels) return dataUrl;
  const scale = Math.sqrt(maxPixels / total);
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return dataUrl;
  ctx.drawImage(img, 0, 0, w, h);
  // Use JPEG for ~3-5x smaller payload than PNG; quality 0.9 preserves
  // UI text legibility for the demo.
  return canvas.toDataURL("image/jpeg", 0.9);
}

interface ImageUploadProps {
  image: string | null;
  onImageSelected: (dataUrl: string) => void;
}

export function ImageUpload({ image, onImageSelected }: ImageUploadProps) {
  const [dragover, setDragover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string | undefined;
        if (!dataUrl) return;
        // Downscale very large screenshots before sending to the API.
        // Anything over ~1.5M pixels (1500x1000) provides no accuracy
        // benefit for grounding and adds 5-25s of latency on T4/A100.
        downscaleIfNeeded(dataUrl, 1_500_000)
          .then(onImageSelected)
          .catch(() => onImageSelected(dataUrl));
      };
      reader.readAsDataURL(file);
    },
    [onImageSelected],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragover(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) handleFile(file);
    },
    [handleFile],
  );

  if (image) {
    return (
      <div
        className="relative border border-border rounded-[var(--radius-base)] overflow-hidden cursor-pointer h-full flex items-center justify-center"
        onClick={() => inputRef.current?.click()}
      >
        <img src={image} alt="Uploaded screenshot" className="max-w-full max-h-full object-contain rounded-lg" />
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "border-2 border-dashed border-border-strong rounded-[var(--radius-base)]",
        "p-8 text-center cursor-pointer transition-all relative bg-bg-elevated h-full flex flex-col items-center justify-center",
        dragover && "border-accent bg-accent-soft",
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragover(true);
      }}
      onDragLeave={() => setDragover(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <div className="w-10 h-10 mx-auto mb-2.5 rounded-[var(--radius-base)] bg-bg-card border border-border flex items-center justify-center shadow-sm">
        <Upload className="w-[18px] h-[18px] text-text-secondary" />
      </div>
      <p className="text-sm text-text-secondary">
        <strong className="text-accent font-semibold">Click to upload</strong> or drag & drop
      </p>
      <p className="text-[0.72rem] text-text-tertiary mt-1">PNG, JPG, WEBP — any resolution</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
    </div>
  );
}
