'use client'

import { useState, useCallback, useRef, DragEvent, ChangeEvent } from 'react'

// ─── Tipler ───────────────────────────────────────────────────────────────────

interface PredictResponse {
  prediction: {
    class_name: string
    class_index: number
    confidence: number
    clinical_group: 'Kanser Şüphesi' | 'Normal Doku' | 'Klinik Dışı' | 'Belirsiz'
  }
  all_probabilities: Record<string, number>
  meta: {
    image_size: string
    device: string
    threshold: number
  }
}

// ─── Sabitler ─────────────────────────────────────────────────────────────────

const API_URL      = process.env.NEXT_PUBLIC_API_URL      ?? 'http://localhost:8000/predict'
const FEEDBACK_URL = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL.replace('/predict', '/feedback')
  : 'http://localhost:8000/feedback'

const CLINICAL_CONFIG = {
  'Kanser Şüphesi': {
    bg: 'bg-red-50',
    border: 'border-red-200',
    badge: 'bg-red-100 text-red-800 border border-red-200',
    dot: 'bg-red-500',
    label: 'Kanser Şüphesi',
  },
  'Normal Doku': {
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    badge: 'bg-emerald-100 text-emerald-800 border border-emerald-200',
    dot: 'bg-emerald-500',
    label: 'Normal Doku',
  },
  'Klinik Dışı': {
    bg: 'bg-slate-50',
    border: 'border-slate-200',
    badge: 'bg-slate-100 text-slate-600 border border-slate-200',
    dot: 'bg-slate-400',
    label: 'Klinik Dışı',
  },
  'Belirsiz': {
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    badge: 'bg-amber-100 text-amber-800 border border-amber-200',
    dot: 'bg-amber-400',
    label: 'Belirsiz',
  },
} as const

// ─── Bileşenler ───────────────────────────────────────────────────────────────

function Spinner({ className = '' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
    </svg>
  )
}

function ProbabilityBar({ label, value, isTop }: { label: string; value: number; isTop: boolean }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-0.5">
        <span className={`text-xs font-medium flex items-center gap-1 ${isTop ? 'text-slate-800' : 'text-slate-500'}`}>
          {isTop && <span className="text-blue-500 text-[10px]">▶</span>}
          {label}
        </span>
        <span className={`text-xs tabular-nums font-semibold ${isTop ? 'text-blue-600' : 'text-slate-400'}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            isTop ? 'bg-blue-500' : value >= 5 ? 'bg-slate-300' : 'bg-slate-200'
          }`}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
    </div>
  )
}

// ─── Ana Sayfa ────────────────────────────────────────────────────────────────

const CLASS_NAMES = [
  'Normal', 'Tümör', 'Stroma', 'Lenfosit',
  'Düz Kas', 'Debris', 'Mukosa', 'Adipoz', 'Arka Plan',
]

export default function Home() {
  const [isDragging, setIsDragging]     = useState(false)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageFile, setImageFile]       = useState<File | null>(null)
  const [loading, setLoading]           = useState(false)
  const [result, setResult]             = useState<PredictResponse | null>(null)
  const [error, setError]               = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Feedback state
  const [showFeedback, setShowFeedback]       = useState(false)
  const [selectedLabel, setSelectedLabel]     = useState<number | null>(null)
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [feedbackMsg, setFeedbackMsg]         = useState<string | null>(null)

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Lütfen geçerli bir görüntü dosyası seçin (JPG, PNG, TIFF, BMP).')
      return
    }
    setError(null)
    setResult(null)
    setImageFile(file)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    // Sadece zone dışına çıkınca tetikle (child elemanlar hariç)
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false)
    }
  }, [])

  const handleInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handlePredict = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', imageFile)
      const res = await fetch(API_URL, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('API\'ye ulaşılamıyor. FastAPI sunucusunun çalıştığından emin olun: uvicorn api:app --reload')
      } else {
        setError(err instanceof Error ? err.message : 'Bilinmeyen hata.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImagePreview(null)
    setImageFile(null)
    setResult(null)
    setError(null)
    setShowFeedback(false)
    setSelectedLabel(null)
    setFeedbackMsg(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleFeedback = async () => {
    if (!imageFile || selectedLabel === null) return
    setFeedbackLoading(true)
    setFeedbackMsg(null)
    try {
      const form = new FormData()
      form.append('file', imageFile)
      form.append('correct_label', String(selectedLabel))
      const res = await fetch(FEEDBACK_URL, { method: 'POST', body: form })
      const data = await res.json()
      if (data.status === 'fine_tune_başlatıldı') {
        setFeedbackMsg(`✅ Kaydedildi! ${data.pending} örnek doldu — model güncelleniyor 🔄`)
      } else {
        setFeedbackMsg(`✅ Kaydedildi. (${data.pending}/${data.threshold} — ${data.remaining} tane daha gerekli)`)
      }
      setShowFeedback(false)
      setSelectedLabel(null)
    } catch {
      setFeedbackMsg('❌ Gönderim başarısız, tekrar dene.')
    } finally {
      setFeedbackLoading(false)
    }
  }

  const clinicalCfg = result
    ? (CLINICAL_CONFIG[result.prediction.clinical_group] ?? CLINICAL_CONFIG['Belirsiz'])
    : null

  const sortedProbs = result
    ? Object.entries(result.all_probabilities).sort(([, a], [, b]) => b - a)
    : []

  return (
    <main className="min-h-screen bg-slate-100 py-10 px-4">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* ── Başlık ── */}
        <header className="text-center">
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
            Histopatoloji Görüntü Sınıflandırma
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Kolon kanseri doku analizi · 9 sınıf · SimpleCancerNet
          </p>
        </header>

        {/* ── İki kolon ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

          {/* ── Sol: Yükleme ── */}
          <div className="flex flex-col gap-4">

            {/* Drag & drop zone */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`
                relative flex flex-col items-center justify-center gap-2
                rounded-xl border-2 border-dashed cursor-pointer select-none
                transition-all duration-200 min-h-[172px] px-6 py-8
                ${isDragging
                  ? 'border-blue-400 bg-blue-50 shadow-inner scale-[1.01]'
                  : 'border-slate-300 bg-white hover:border-blue-300 hover:bg-slate-50'}
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleInputChange}
              />
              <div className="text-4xl">🔬</div>
              <p className="text-sm font-medium text-slate-600">
                Görüntüyü buraya sürükle &amp; bırak
              </p>
              <p className="text-xs text-slate-400">veya tıkla ve seç</p>
              <p className="text-[11px] text-slate-300 mt-1">JPG · PNG · TIFF · BMP</p>
            </div>

            {/* Önizleme */}
            {imagePreview && (
              <div className="rounded-xl overflow-hidden bg-white border border-slate-200 shadow-sm">
                <img
                  src={imagePreview}
                  alt="Yüklenen görüntü"
                  className="w-full object-contain max-h-56"
                />
                {imageFile && (
                  <div className="flex justify-between px-3 py-2 text-[11px] text-slate-400 border-t border-slate-100">
                    <span className="truncate max-w-[70%]">{imageFile.name}</span>
                    <span>{(imageFile.size / 1024).toFixed(0)} KB</span>
                  </div>
                )}
              </div>
            )}

            {/* Butonlar */}
            <div className="flex gap-3">
              <button
                onClick={handlePredict}
                disabled={!imageFile || loading}
                className="
                  flex-1 flex items-center justify-center gap-2
                  py-2.5 rounded-lg text-sm font-semibold
                  bg-blue-600 text-white
                  hover:bg-blue-700 active:bg-blue-800
                  disabled:opacity-40 disabled:cursor-not-allowed
                  transition-colors duration-150
                "
              >
                {loading ? (
                  <>
                    <Spinner className="h-4 w-4" />
                    Analiz ediliyor...
                  </>
                ) : 'Analiz Et'}
              </button>

              {(imagePreview || result) && (
                <button
                  onClick={handleReset}
                  className="
                    px-4 py-2.5 rounded-lg text-sm font-semibold
                    bg-slate-200 text-slate-600
                    hover:bg-slate-300 active:bg-slate-400
                    transition-colors duration-150
                  "
                >
                  Temizle
                </button>
              )}
            </div>

            {/* Hata */}
            {error && (
              <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700 leading-relaxed">
                <span className="font-semibold">Hata: </span>{error}
              </div>
            )}
          </div>

          {/* ── Sağ: Sonuçlar ── */}
          <div className="flex flex-col gap-4">

            {/* Boş durum */}
            {!result && !loading && (
              <div className="flex-1 flex items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white min-h-[300px]">
                <p className="text-sm text-slate-400">Sonuçlar burada görünecek</p>
              </div>
            )}

            {/* Yükleniyor */}
            {loading && (
              <div className="flex-1 flex flex-col items-center justify-center rounded-xl bg-white border border-slate-200 min-h-[300px] gap-3">
                <Spinner className="h-9 w-9 text-blue-500" />
                <p className="text-sm text-slate-500">Model çalışıyor...</p>
              </div>
            )}

            {/* Sonuç kartı */}
            {result && clinicalCfg && (
              <>
                {/* Tahmin özeti */}
                <div className={`rounded-xl border p-4 ${clinicalCfg.bg} ${clinicalCfg.border}`}>
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <div>
                      <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-1">
                        Tahmin
                      </p>
                      <p className="text-2xl font-bold text-slate-800">
                        {result.prediction.class_name}
                      </p>
                      <p className="text-sm text-slate-600 mt-0.5">
                        Güven:&nbsp;
                        <span className="font-bold text-slate-800">
                          {result.prediction.confidence}%
                        </span>
                      </p>
                    </div>

                    <div className="flex flex-col items-end gap-2 shrink-0">
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${clinicalCfg.badge}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${clinicalCfg.dot}`} />
                        {clinicalCfg.label}
                      </span>
                      <span className="text-[11px] text-slate-400">
                        {result.meta.device.toUpperCase()} · {result.meta.image_size}
                      </span>
                    </div>
                  </div>

                  {/* Güven çubuğu */}
                  <div className="h-2 bg-white/70 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-700 ease-out"
                      style={{ width: `${result.prediction.confidence}%` }}
                    />
                  </div>
                  <div className="flex justify-between mt-1 text-[10px] text-slate-400">
                    <span>0%</span>
                    <span>Eşik: {result.meta.threshold * 100}%</span>
                    <span>100%</span>
                  </div>
                </div>

                {/* Olasılık bar chart */}
                <div className="rounded-xl bg-white border border-slate-200 p-4">
                  <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-3">
                    Sınıf Olasılıkları
                  </p>
                  <div className="space-y-2.5">
                    {sortedProbs.map(([cls, prob]) => (
                      <ProbabilityBar
                        key={cls}
                        label={cls}
                        value={prob}
                        isTop={cls === result.prediction.class_name}
                      />
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* ── Feedback ── */}
            {result && (
              <div className="rounded-xl bg-white border border-slate-200 p-4 space-y-3">
                {!showFeedback ? (
                  <button
                    onClick={() => setShowFeedback(true)}
                    className="w-full text-sm text-slate-500 hover:text-slate-700 underline underline-offset-2 transition-colors"
                  >
                    Tahmin yanlış mı? Düzelt
                  </button>
                ) : (
                  <>
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                      Doğru sınıf nedir?
                    </p>
                    <div className="grid grid-cols-3 gap-1.5">
                      {CLASS_NAMES.map((name, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedLabel(idx)}
                          className={`
                            px-2 py-1.5 rounded-lg text-xs font-medium border transition-all
                            ${selectedLabel === idx
                              ? 'bg-blue-600 text-white border-blue-600'
                              : 'bg-slate-50 text-slate-600 border-slate-200 hover:border-blue-300'}
                          `}
                        >
                          {name}
                        </button>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleFeedback}
                        disabled={selectedLabel === null || feedbackLoading}
                        className="flex-1 py-2 rounded-lg text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {feedbackLoading ? 'Gönderiliyor…' : 'Gönder'}
                      </button>
                      <button
                        onClick={() => { setShowFeedback(false); setSelectedLabel(null) }}
                        className="px-4 py-2 rounded-lg text-sm font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
                      >
                        İptal
                      </button>
                    </div>
                  </>
                )}
                {feedbackMsg && (
                  <p className="text-xs text-center text-slate-500">{feedbackMsg}</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
