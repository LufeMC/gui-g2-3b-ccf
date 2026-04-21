import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

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
        if (e.target?.result) onImageSelected(e.target.result as string);
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
