"use client"

import { useState, useRef, useCallback, DragEvent, ChangeEvent } from "react"
import { motion } from "framer-motion"
import { Upload, X, Search, Loader2, AlertCircle, User } from "lucide-react"

type SearchMode = "CNN + Features" | "CNN Only" | "Features Only"

interface Match {
  name: string
  similarity: number
  image?: string
}

function parseGradioResponse(data: unknown): Match[] {
  if (!data || typeof data !== "object") return []
  const d = data as Record<string, unknown>

  // data[0] is the gallery result
  const raw = Array.isArray(d.data) ? d.data[0] : null
  if (!raw) return []

  // If it's an array of {image, caption} pairs (Gradio gallery format)
  if (Array.isArray(raw)) {
    return raw.slice(0, 5).map((item: unknown, i: number) => {
      if (item && typeof item === "object") {
        const obj = item as Record<string, unknown>
        const caption = typeof obj.caption === "string" ? obj.caption : `Match ${i + 1}`
        const imgData = obj.image ?? obj.url ?? null
        const imageStr = typeof imgData === "string" ? imgData : (imgData && typeof imgData === "object" ? (imgData as Record<string, unknown>).url as string : undefined)

        // Parse similarity from caption like "Tom Hanks (87.3%)" or "87.3% - Tom Hanks"
        const pctMatch = caption.match(/(\d+\.?\d*)%/)
        const similarity = pctMatch ? parseFloat(pctMatch[1]) : Math.max(95 - i * 8, 40)

        // Parse name
        const nameMatch = caption.match(/^(.+?)\s*[\(\-]/)
        const name = nameMatch ? nameMatch[1].trim() : caption.replace(/[\d.%\(\)\-]/g, "").trim() || `Match ${i + 1}`

        return { name, similarity, image: imageStr }
      }
      return { name: `Match ${i + 1}`, similarity: Math.max(90 - i * 10, 30) }
    })
  }

  // If it's a string (text output)
  if (typeof raw === "string") {
    const lines = raw.split("\n").filter(Boolean).slice(0, 5)
    return lines.map((line, i) => {
      const pctMatch = line.match(/(\d+\.?\d*)%/)
      const similarity = pctMatch ? parseFloat(pctMatch[1]) : Math.max(90 - i * 10, 30)
      const name = line.replace(/[\d.%\(\)\-:]/g, "").trim() || `Match ${i + 1}`
      return { name, similarity }
    })
  }

  return []
}

export function CelebrityFinder() {
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [mode, setMode] = useState<SearchMode>("CNN + Features")
  const [loading, setLoading] = useState(false)
  const [matches, setMatches] = useState<Match[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file.")
      return
    }
    const reader = new FileReader()
    reader.onload = (e) => {
      setImageDataUrl(e.target?.result as string)
      setMatches(null)
      setError(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) loadFile(file)
    },
    [loadFile]
  )

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) loadFile(file)
  }

  const clearImage = () => {
    setImageDataUrl(null)
    setMatches(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleSearch = async () => {
    if (!imageDataUrl) return
    setLoading(true)
    setError(null)
    setMatches(null)

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl, mode }),
      })

      const json = await res.json()

      if (!res.ok) {
        throw new Error(json.error || "Search failed")
      }

      const parsed = parseGradioResponse(json)
      if (parsed.length === 0) {
        setError("No matches found. Try a different photo or mode.")
      } else {
        setMatches(parsed)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const modes: SearchMode[] = ["CNN + Features", "CNN Only", "Features Only"]

  return (
    <section id="finder" className="py-24 md:py-32 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-12 text-center"
        >
          <span className="inline-block text-xs font-semibold tracking-widest text-(--ollie-cyan) uppercase mb-3">
            Core Feature
          </span>
          <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight">
            Find Your Celebrity Match
          </h2>
          <p className="mt-4 text-white/50 max-w-xl mx-auto">
            Drop your photo. Choose a search mode. See which celebrity you resemble.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 items-start">
          {/* LEFT: Uploader */}
          <motion.div
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="flex flex-col gap-5"
          >
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => !imageDataUrl && fileInputRef.current?.click()}
              className={`relative rounded-2xl border-2 border-dashed transition-all cursor-pointer
                ${isDragging
                  ? "border-(--ollie-cyan) bg-(--ollie-glow)"
                  : imageDataUrl
                    ? "border-white/10 bg-white/[0.02]"
                    : "border-white/15 bg-white/[0.02] hover:border-white/30 hover:bg-white/[0.03]"
                }`}
              style={{ minHeight: 280 }}
            >
              {imageDataUrl ? (
                <div className="relative w-full h-full">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageDataUrl}
                    alt="Uploaded preview"
                    className="w-full rounded-2xl object-cover"
                    style={{ maxHeight: 380, objectFit: "cover" }}
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); clearImage() }}
                    className="absolute top-3 right-3 p-1.5 rounded-full bg-black/70 border border-white/10 text-white hover:bg-black/90 transition-colors"
                    aria-label="Remove image"
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8">
                  <div className="p-4 rounded-full bg-white/5 border border-white/10">
                    <Upload size={28} className="text-white/40" />
                  </div>
                  <div className="text-center">
                    <p className="text-white/70 font-medium">Drag &amp; drop your photo</p>
                    <p className="text-white/30 text-sm mt-1">or click to browse</p>
                  </div>
                  <span className="text-white/20 text-xs">JPG, PNG, WEBP Â· Max 10 MB</span>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onFileChange}
              />
            </div>

            {/* Mode selector */}
            <div>
              <p className="text-white/40 text-xs font-semibold uppercase tracking-wider mb-3">Search Mode</p>
              <div className="flex gap-2">
                {modes.map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex-1 py-2 px-2 text-xs font-semibold rounded-xl border transition-all ${
                      mode === m
                        ? "border-(--ollie-cyan) bg-(--ollie-cyan)/10 text-(--ollie-cyan)"
                        : "border-white/10 bg-white/[0.02] text-white/40 hover:border-white/20 hover:text-white/60"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            {/* Search button */}
            <button
              onClick={handleSearch}
              disabled={!imageDataUrl || loading}
              className={`flex items-center justify-center gap-2 w-full py-3.5 rounded-xl font-bold text-sm transition-all
                ${imageDataUrl && !loading
                  ? "bg-(--ollie-cyan) text-black hover:opacity-90 active:scale-[0.98] shadow-lg shadow-(--ollie-glow)"
                  : "bg-white/5 text-white/20 cursor-not-allowed"
                }`}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Search
                </>
              )}
            </button>
          </motion.div>

          {/* RIGHT: Results */}
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="flex flex-col gap-4"
          >
            <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-5" style={{ minHeight: 380 }}>
              {/* Placeholder */}
              {!loading && !matches && !error && (
                <div className="h-full flex flex-col items-center justify-center gap-4 py-16 text-center">
                  <div className="p-4 rounded-full bg-white/5 border border-white/5">
                    <User size={32} className="text-white/20" />
                  </div>
                  <div>
                    <p className="text-white/30 font-medium">Results will appear here</p>
                    <p className="text-white/15 text-sm mt-1">Upload a photo and hit Search</p>
                  </div>
                </div>
              )}

              {/* Loading */}
              {loading && (
                <div className="h-full flex flex-col items-center justify-center gap-4 py-16">
                  <Loader2 size={36} className="text-(--ollie-cyan) animate-spin" />
                  <p className="text-white/40 text-sm">Scanning faces...</p>
                </div>
              )}

              {/* Error */}
              {error && !loading && (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                  <AlertCircle size={18} className="text-red-400 shrink-0 mt-0.5" />
                  <p className="text-red-300 text-sm leading-relaxed">{error}</p>
                </div>
              )}

              {/* Results */}
              {matches && !loading && (
                <div className="flex flex-col gap-3">
                  <p className="text-white/40 text-xs font-semibold uppercase tracking-wider mb-1">
                    Top {matches.length} Matches
                  </p>
                  {matches.map((match, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.07 }}
                      className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.03] border border-white/5"
                    >
                      {/* Thumbnail */}
                      <div className="w-11 h-11 rounded-lg overflow-hidden bg-white/5 shrink-0 flex items-center justify-center border border-white/10">
                        {match.image ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={match.image} alt={match.name} className="w-full h-full object-cover" />
                        ) : (
                          <User size={18} className="text-white/20" />
                        )}
                      </div>

                      {/* Name + bar */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-white text-sm font-semibold truncate">{match.name}</span>
                          <span className="text-(--ollie-cyan) text-xs font-bold ml-2 shrink-0">
                            {match.similarity.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(match.similarity, 100)}%` }}
                            transition={{ duration: 0.8, delay: i * 0.1 + 0.2, ease: "easeOut" }}
                            className="h-full rounded-full bg-(--ollie-cyan)"
                          />
                        </div>
                      </div>

                      {/* Rank badge */}
                      {i === 0 && (
                        <span className="shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded bg-(--ollie-cyan)/15 text-(--ollie-cyan) border border-(--ollie-cyan)/30">
                          #1
                        </span>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
