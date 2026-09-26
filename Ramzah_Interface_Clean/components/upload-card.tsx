"use client"

import { useRef } from "react"
import { Camera, Upload, Video } from "lucide-react"

interface UploadCardProps {
  onNext: (file: File) => void
  busy?: boolean
  error?: string
}

export function UploadCard({ onNext, busy = false, error = "" }: UploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const cameraRef = useRef<HTMLInputElement>(null)

  const selected = (file?: File) => {
    if (file && !busy) onNext(file)
  }

  return (
    <section dir="rtl" className="flex h-full min-h-0 flex-col rounded-[30px] bg-white px-8 py-9 sm:px-10 lg:px-12 lg:py-12">
      <div className="flex flex-1 flex-col justify-center">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex min-h-[350px] flex-1 flex-col items-center justify-center gap-5 rounded-[18px] border-2 border-dashed border-black/70 px-6 text-center text-black transition-colors hover:bg-neutral-50 md:max-h-[410px]"
        >
          <Camera className="size-10 text-black" strokeWidth={1.65} />
          <p className="text-[17px] font-normal leading-8 text-black sm:text-[18px]">{busy ? "جارٍ تحليل الإشارة وصياغة الجملة..." : "ارفع فيديو أو ابدأ التصوير المباشر"}</p>
        </button>

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => { selected(e.target.files?.[0]); e.target.value = "" }}
        />
        <input ref={cameraRef} type="file" accept="video/*" capture="environment" className="hidden" onChange={(e) => { selected(e.target.files?.[0]); e.target.value = "" }} />

        {error && <p role="alert" className="mt-3 text-center text-sm text-red-600">{error}</p>}

        <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => cameraRef.current?.click()}
            disabled={busy}
            className="flex items-center justify-center gap-3 rounded-[10px] border border-black/75 px-5 py-3.5 text-[16px] font-medium text-black transition-colors hover:bg-neutral-50"
          >
            <Video className="size-5 text-black" strokeWidth={1.7} />
            تشغيل الكاميرا
          </button>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
            className="flex items-center justify-center gap-3 rounded-[10px] border border-black/75 px-5 py-3.5 text-[16px] font-medium text-black transition-colors hover:bg-neutral-50"
          >
            <Upload className="size-5 text-black" strokeWidth={1.7} />
            رفع فيديو
          </button>
        </div>
      </div>
    </section>
  )
}
