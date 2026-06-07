"use client"

import { useState, useRef, useCallback, ChangeEvent, DragEvent } from "react"
import { motion } from "framer-motion"
import { ThumbsUp, ThumbsDown, Upload, X, Loader2, CheckCircle, AlertCircle } from "lucide-react"

function MiniUploader({
  label,
  value,
  onChange,
}: {
  label: string
  value: string | null
  onChange: (dataUrl: string | null) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const loadFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return
      const reader = new FileReader()
      reader.onload = (e) => onChange(e.target?.result as string)
      reader.readAsDataURL(file)
    },
    [onChange]
  )

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) loadFile(file)
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-white/40 text-xs font-semibold uppercase tracking-wider">{label}</p>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !value && inputRef.current?.click()}
        className={`relative rounded-xl border-2 border-dashed transition-all cursor-pointer flex items-center justify-center
          ${dragging ? "border-(--ollie-cyan) bg-(--ollie-glow)" : value ? "border-white/10 bg-white/[0.02]" : "border-white/15 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.03]"}
        `}
        style={{ height: 140 }}
      >
        {value ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={value} alt={label} className="w-full h-full object-cover rounded-xl" />
            <button
              onClick={(e) => { e.stopPropagation(); onChange(null) }}
              className="absolute top-2 right-2 p-1 rounded-full bg-black/70 border border-white/10 text-white hover:bg-black/90 transition-colors"
            >
              <X size={12} />
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 text-center px-4">
            <Upload size={20} className="text-white/25" />
            <span className="text-white/30 text-xs">Drop or click</span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0]
            if (file) loadFile(file)
          }}
        />
      </div>
    </div>
  )
}

export function FeedbackSection() {
  const [photo1, setPhoto1] = useState<string | null>(null)
  const [photo2, setPhoto2] = useState<string | null>(null)
  const [isMatch, setIsMatch] = useState<boolean | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!photo1 || !photo2 || isMatch === null) return
    setSubmitting(true)
    setError(null)

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image1: photo1,
          image2: photo2,
          label: isMatch ? 1 : 0,
        }),
      })

      const json = await res.json()
      if (!res.ok) throw new Error(json.error || "Submission failed")
      setSubmitted(true)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setError(msg)
    } finally {
      setSubmitting(false)
    }
  }

  const reset = () => {
    setPhoto1(null)
    setPhoto2(null)
    setIsMatch(null)
    setSubmitted(false)
    setError(null)
  }

  const canSubmit = photo1 && photo2 && isMatch !== null && !submitting

  return (
    <section id="feedback" className="py-24 md:py-32 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-block text-xs font-semibold tracking-widest text-(--ollie-cyan) uppercase mb-3">
            Improve the Model
          </span>
          <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight">
            Help Ollie Learn
          </h2>
          <p className="mt-4 text-white/50 max-w-xl mx-auto">
            Correct a bad match and improve the model for everyone.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="max-w-xl mx-auto"
        >
          {submitted ? (
            /* Success state */
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center gap-5 py-16 text-center rounded-2xl border border-white/5 bg-white/[0.02] px-8"
            >
              <div className="p-4 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                <CheckCircle size={36} className="text-emerald-400" />
              </div>
              <div>
                <h3 className="text-white font-bold text-lg">Thank you!</h3>
                <p className="text-white/40 text-sm mt-2">
                  Your feedback has been recorded. Every submission helps Ollie get smarter.
                </p>
              </div>
              <button
                onClick={reset}
                className="px-6 py-2.5 rounded-full border border-white/15 text-white/60 hover:text-white hover:border-white/30 text-sm transition-all"
              >
                Submit another
              </button>
            </motion.div>
          ) : (
            <div className="flex flex-col gap-6 p-6 rounded-2xl border border-white/5 bg-white/[0.02]">
              {/* Two uploaders */}
              <div className="grid grid-cols-2 gap-4">
                <MiniUploader label="Your Photo" value={photo1} onChange={setPhoto1} />
                <MiniUploader label="The Person We Matched" value={photo2} onChange={setPhoto2} />
              </div>

              {/* Match toggle */}
              <div>
                <p className="text-white/40 text-xs font-semibold uppercase tracking-wider mb-3">
                  Is this a match?
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setIsMatch(true)}
                    className={`flex items-center justify-center gap-2 py-4 rounded-xl border font-semibold text-sm transition-all
                      ${isMatch === true
                        ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-400"
                        : "border-white/10 bg-white/[0.02] text-white/40 hover:border-white/20 hover:text-white/60"
                      }`}
                  >
                    <ThumbsUp size={18} />
                    Yes, it matches
                  </button>
                  <button
                    onClick={() => setIsMatch(false)}
                    className={`flex items-center justify-center gap-2 py-4 rounded-xl border font-semibold text-sm transition-all
                      ${isMatch === false
                        ? "border-red-500/60 bg-red-500/10 text-red-400"
                        : "border-white/10 bg-white/[0.02] text-white/40 hover:border-white/20 hover:text-white/60"
                      }`}
                  >
                    <ThumbsDown size={18} />
                    No, wrong match
                  </button>
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="flex items-start gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20">
                  <AlertCircle size={16} className="text-red-400 shrink-0 mt-0.5" />
                  <p className="text-red-300 text-sm">{error}</p>
                </div>
              )}

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className={`flex items-center justify-center gap-2 w-full py-3.5 rounded-xl font-bold text-sm transition-all
                  ${canSubmit
                    ? "bg-(--ollie-cyan) text-black hover:opacity-90 active:scale-[0.98] shadow-lg shadow-(--ollie-glow)"
                    : "bg-white/5 text-white/20 cursor-not-allowed"
                  }`}
              >
                {submitting ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Submitting...
                  </>
                ) : (
                  "Submit Feedback"
                )}
              </button>

              <p className="text-white/20 text-xs text-center">
                Your images are processed locally and never stored.
              </p>
            </div>
          )}
        </motion.div>
      </div>
    </section>
  )
}
