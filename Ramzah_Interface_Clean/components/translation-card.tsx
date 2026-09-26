"use client"

import { useState } from "react"
import { ArrowLeft, Send } from "lucide-react"

interface TranslationCardProps {
  translation: string
  onBack: () => void
  onRetry: () => void
  onConfirm: () => Promise<void>
  busy?: boolean
  error?: string
}

export function TranslationCard({
  translation,
  onBack,
  onRetry,
  onConfirm,
  busy = false,
  error = "",
}: TranslationCardProps) {
  const [reply, setReply] = useState("")

  return (
    <section dir="rtl" className="flex h-full min-h-0 flex-col overflow-hidden rounded-[30px] bg-white px-8 py-8 sm:px-10 lg:px-12 lg:py-9">
      <div dir="ltr" className="flex shrink-0 items-center">
        <button type="button" aria-label="رجوع" onClick={onBack} className="flex size-9 items-center justify-center rounded-full bg-neutral-100 text-neutral-700">
          <ArrowLeft className="size-4" />
        </button>
      </div>

      <div className="mt-4 flex shrink-0 flex-col">
        <h2 className="mb-3 text-right text-[22px] font-medium text-neutral-900">ترجمة الإشارة</h2>
        <div className="flex min-h-[190px] items-center justify-center rounded-md border border-neutral-400 px-6 py-7 text-center md:min-h-[210px]">
          <p className="text-[21px] font-normal text-neutral-900 sm:text-[22px]">{translation}</p>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <button type="button" onClick={onRetry} disabled={busy} className="rounded-md border border-neutral-400 px-5 py-3.5 text-[16px] font-normal text-neutral-900">إعادة المحاولة</button>
          <button type="button" onClick={onConfirm} disabled={busy} className="rounded-md border border-neutral-400 px-5 py-3.5 text-[16px] font-normal text-neutral-900">{busy ? "جارٍ تجهيز الصوت..." : "تأكيد ونطق"}</button>
        </div>
        {error && <p role="alert" className="mt-3 text-center text-sm text-red-600">{error}</p>}
      </div>

      <hr className="my-6 shrink-0 border-neutral-200" />

      <div className="flex min-h-0 flex-1 flex-col">
        <h2 className="mb-3 shrink-0 text-right text-[21px] font-medium text-neutral-900">رد الموظف</h2>
        <div className="flex min-h-[78px] items-center gap-2 rounded-md border border-neutral-400 px-3 py-2">
          <button type="button" aria-label="إرسال" className="flex size-10 shrink-0 items-center justify-center rounded-md border border-neutral-400 text-neutral-800">
            <Send className="size-5 -scale-x-100" />
          </button>
          <input value={reply} onChange={(e) => setReply(e.target.value)} placeholder="اكتب الرد هنا..." className="flex-1 bg-transparent py-2 text-right text-[15px] text-neutral-800 placeholder:text-neutral-400 focus:outline-none" />
        </div>
      </div>
    </section>
  )
}
