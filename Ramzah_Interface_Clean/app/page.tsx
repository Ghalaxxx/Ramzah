"use client"

import { useState } from "react"
import { UploadCard } from "@/components/upload-card"
import { TranslationCard } from "@/components/translation-card"
import { BrandPanel } from "@/components/brand-panel"

export default function Page() {
  const [page, setPage] = useState<"upload" | "translation">("upload")
  const [sessionId, setSessionId] = useState("")
  const [translation, setTranslation] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const request = async (path: string, init?: RequestInit) => {
    const response = await fetch(`/api${path}`, init)
    const data = await response.json().catch(() => ({ detail: "تعذر الاتصال بالخادم" }))
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "تعذر إكمال الطلب")
    return data
  }

  const translateVideo = async (file: File) => {
    setBusy(true); setError("")
    try {
      const session = await request("/sessions", { method: "POST" })
      const form = new FormData(); form.append("file", file)
      const recognized = await request(`/sessions/${session.session_id}/signs`, { method: "POST", body: form })
      let glosses: string[] = recognized.glosses || []
      if (!glosses.length && recognized.prediction) {
        const reviewed = await request(`/sessions/${session.session_id}/glosses`, {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ glosses: [recognized.prediction] }),
        })
        glosses = reviewed.glosses
      }
      if (!glosses.length) throw new Error("لم يتمكن نموذج الرؤية من التعرف على الإشارة")
      const generated = await request(`/sessions/${session.session_id}/sentences`, { method: "POST" })
      if (!generated.candidates?.length) throw new Error("لم يتمكن النموذج اللغوي من توليد جملة")
      setSessionId(session.session_id); setTranslation(generated.candidates[0]); setPage("translation")
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "حدث خطأ غير متوقع")
    } finally { setBusy(false) }
  }

  const confirmAndSpeak = async () => {
    if (!sessionId || !translation) return
    setBusy(true); setError("")
    try {
      await request(`/sessions/${sessionId}/confirm`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: translation }),
      })
      const response = await fetch(`/api/sessions/${sessionId}/speech.wav`)
      if (!response.ok) throw new Error("تعذر إنشاء الصوت")
      const audio = new Audio(URL.createObjectURL(await response.blob()))
      await audio.play()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "حدث خطأ غير متوقع")
    } finally { setBusy(false) }
  }

  const reset = () => { setPage("upload"); setSessionId(""); setTranslation(""); setError("") }

  return (
    <main className="min-h-screen w-full bg-white px-4 py-5 sm:px-6 sm:py-7 lg:px-8 lg:py-8">
      <div
        dir="ltr"
        className="mx-auto grid w-full max-w-[1240px] overflow-hidden rounded-[42px] bg-[#08383b] p-[18px] shadow-sm md:h-[min(780px,calc(100vh-64px))] md:min-h-[650px] md:grid-cols-[1.05fr_0.95fr] md:gap-0 lg:p-[20px]"
      >
        <div className="min-h-0 h-full">
          {page === "upload" ? (
            <UploadCard onNext={translateVideo} busy={busy} error={error} />
          ) : (
            <TranslationCard translation={translation} onBack={reset} onRetry={reset} onConfirm={confirmAndSpeak} busy={busy} error={error} />
          )}
        </div>
        <BrandPanel />
      </div>
    </main>
  )
}
