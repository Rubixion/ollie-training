"use client"

import { motion } from "framer-motion"

const ENTRIES = [
  {
    date: "Jun 2026",
    version: "v2.0",
    title: "VGGFace2 + GPU Features",
    desc: "Added 3.3M training images, InsightFace GPU extraction, age/gender penalties. Major quality jump.",
    badge: "LATEST",
    accent: "cyan",
  },
  {
    date: "Jun 2026",
    version: "v1.5",
    title: "FAISS Search + 3 Modes",
    desc: "100x faster search with FAISS. Added CNN Only, Features Only, and CNN+Features modes.",
    badge: null,
    accent: "purple",
  },
  {
    date: "May 2026",
    version: "v1.0",
    title: "Initial Release",
    desc: "Siamese network trained on LFW. MediaPipe feature extraction. First working prototype.",
    badge: null,
    accent: "white",
  },
]

export function ChangelogSection() {
  return (
    <section className="py-24 md:py-32 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <h2 className="retro text-lg md:text-xl text-white mb-2">CHANGELOG</h2>
          <p className="retro text-[10px] text-white/30 tracking-widest">VERSION HISTORY</p>
        </motion.div>

        <div className="relative max-w-2xl mx-auto">
          {/* Timeline line */}
          <div className="absolute left-[18px] top-3 bottom-3 w-px bg-white/8" />

          <div className="flex flex-col gap-10">
            {ENTRIES.map((entry, i) => {
              const accentColor =
                entry.accent === "cyan"
                  ? "var(--ollie-cyan)"
                  : entry.accent === "purple"
                    ? "var(--ollie-purple)"
                    : "oklch(1 0 0 / 40%)"

              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                  className="flex gap-6 pl-10 relative"
                >
                  {/* Timeline dot */}
                  <div
                    className="absolute left-[12px] top-1 w-3 h-3 rounded-sm rotate-45 shrink-0"
                    style={{
                      background: accentColor,
                      boxShadow: `0 0 10px ${accentColor}`,
                    }}
                  />

                  <div className="flex-1">
                    {/* Date + version + badge */}
                    <div className="flex items-center flex-wrap gap-3 mb-2">
                      <span className="retro text-[9px] text-white/30">{entry.date}</span>
                      <span
                        className="retro text-[9px] px-2 py-0.5 border"
                        style={{ color: accentColor, borderColor: accentColor, background: `${accentColor}15` }}
                      >
                        {entry.version}
                      </span>
                      {entry.badge && (
                        <span className="retro text-[8px] px-2 py-0.5 bg-(--ollie-cyan) text-black">
                          {entry.badge}
                        </span>
                      )}
                    </div>

                    {/* Title */}
                    <h3 className="retro text-[11px] text-white mb-2 leading-loose">{entry.title}</h3>

                    {/* Desc */}
                    <p className="text-white/40 text-sm leading-relaxed font-sans">{entry.desc}</p>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </div>
    </section>
  )
}
