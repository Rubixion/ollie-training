"use client"

import { motion } from "framer-motion"
import { Lock } from "lucide-react"

export function FindPerson() {
  return (
    <section id="find-person" className="py-24 md:py-32 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-(--ollie-purple)/40 bg-(--ollie-purple)/10 text-(--ollie-purple) text-xs font-semibold tracking-wide mb-4">
            Coming Soon
          </span>
          <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight">
            Find Anyone, Anywhere
          </h2>
          <p className="mt-4 text-white/50 max-w-xl mx-auto">
            Same model, infinite scale. Search across the internet for a face. Not available yet.
          </p>
        </motion.div>

        {/* Locked UI */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="relative rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden"
        >
          {/* Blurred content */}
          <div className="opacity-30 pointer-events-none select-none p-8">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Fake uploader */}
              <div className="rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] flex flex-col items-center justify-center gap-4 py-16">
                <div className="p-4 rounded-full bg-white/5 border border-white/10">
                  <div className="w-8 h-8 rounded bg-white/20" />
                </div>
                <div className="text-center">
                  <p className="text-white/70 font-medium">Upload a photo</p>
                  <p className="text-white/30 text-sm mt-1">or paste an image URL</p>
                </div>
              </div>

              {/* Fake results */}
              <div className="flex flex-col gap-3">
                <div className="h-10 rounded-xl bg-white/5" />
                {[88, 74, 61, 52, 43].map((w, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.03] border border-white/5">
                    <div className="w-11 h-11 rounded-lg bg-white/10 shrink-0" />
                    <div className="flex-1 flex flex-col gap-2">
                      <div className="h-3 rounded bg-white/10" style={{ width: `${w}%` }} />
                      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                        <div className="h-full rounded-full bg-white/20" style={{ width: `${w}%` }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Fake internet search options */}
            <div className="mt-6 flex gap-3">
              {["Google Images", "Instagram", "LinkedIn", "Twitter / X"].map((s) => (
                <div key={s} className="px-4 py-2 rounded-xl border border-white/10 bg-white/5 text-white/40 text-sm">
                  {s}
                </div>
              ))}
            </div>
          </div>

          {/* Coming Soon overlay */}
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-6">
            {/* Pulsing lock */}
            <motion.div
              animate={{
                scale: [1, 1.08, 1],
                boxShadow: [
                  "0 0 0px oklch(0.65 0.22 290 / 0%)",
                  "0 0 40px oklch(0.65 0.22 290 / 30%)",
                  "0 0 0px oklch(0.65 0.22 290 / 0%)",
                ],
              }}
              transition={{ repeat: Infinity, duration: 2.5, ease: "easeInOut" }}
              className="p-5 rounded-2xl bg-(--ollie-purple)/10 border border-(--ollie-purple)/30"
            >
              <Lock size={36} className="text-(--ollie-purple)" />
            </motion.div>

            {/* 8bit COMING SOON badge */}
            <div className="text-center">
              <p
                className="retro text-(--ollie-purple) text-sm md:text-base"
                style={{ textShadow: "0 0 20px oklch(0.65 0.22 290 / 60%)" }}
              >
                COMING SOON
              </p>
              <p className="text-white/30 text-sm mt-3 font-sans">
                Internet-scale face search is in active development.
              </p>
            </div>

            {/* Progress indicator */}
            <div className="flex flex-col items-center gap-2">
              <div className="flex gap-1.5">
                {[1, 2, 3, 4, 5].map((i) => (
                  <motion.div
                    key={i}
                    animate={{ opacity: [0.2, 1, 0.2] }}
                    transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.2 }}
                    className={`rounded-full ${i <= 2 ? "bg-(--ollie-purple)" : "bg-white/15"}`}
                    style={{ width: 8, height: 8 }}
                  />
                ))}
              </div>
              <p className="text-white/20 text-xs">2 / 5 milestones complete</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
