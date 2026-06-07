"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, Minus, ArrowRight } from "lucide-react"
import Link from "next/link"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"

// â”€â”€ FAQ data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const FAQ_ITEMS = [
  {
    q: "WHAT IS OLLIE?",
    a: "Ollie uses a custom-trained Siamese Neural Network to find faces that look like yours in a curated dataset.",
  },
  {
    q: "HOW ACCURATE IS IT?",
    a: "The model is trained on LFW and VGGFace2 datasets. Accuracy improves continuously with user feedback.",
  },
  {
    q: "IS MY PHOTO STORED?",
    a: "No. Your image is processed locally and never stored on our servers.",
  },
  {
    q: "WHAT SEARCH MODES ARE THERE?",
    a: "CNN + Features (best), CNN Only (faster), Features Only (geometric matching only).",
  },
  {
    q: "CAN I SEARCH THE INTERNET?",
    a: "Not yet. Internet search is in development under 'Face Search' â€” check back soon.",
  },
  {
    q: "HOW DO I HELP IMPROVE IT?",
    a: "Use the Feedback section to correct bad matches. Every submission makes the model smarter.",
  },
]

// â”€â”€ Changelog data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const ENTRIES = [
  {
    date: "Jun 2026",
    version: "v2.0",
    title: "VGGFace2 + GPU Features",
    desc: "Added 3.3M training images, InsightFace GPU extraction, age/gender penalties. Major quality jump.",
    badge: "LATEST",
    accent: "cyan" as const,
  },
  {
    date: "Jun 2026",
    version: "v1.5",
    title: "FAISS Search + 3 Modes",
    desc: "100x faster search with FAISS. Added CNN Only, Features Only, and CNN+Features modes.",
    badge: null,
    accent: "purple" as const,
  },
  {
    date: "May 2026",
    version: "v1.0",
    title: "Initial Release",
    desc: "Siamese network trained on LFW. MediaPipe feature extraction. First working prototype.",
    badge: null,
    accent: "white" as const,
  },
]

// â”€â”€ Accordion item â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function AccordionItem({
  item,
  isOpen,
  onToggle,
  index,
}: {
  item: { q: string; a: string }
  isOpen: boolean
  onToggle: () => void
  index: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.07 }}
      style={{
        borderLeft: "3px solid",
        borderLeftColor: isOpen ? "var(--ollie-cyan)" : "oklch(1 0 0 / 8%)",
        background: isOpen ? "oklch(1 0 0 / 2%)" : "transparent",
        transition: "border-left-color 0.2s, background 0.2s",
      }}
      className="border border-white/8 rounded-none"
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-5 py-4 text-left group"
      >
        <span
          className={`retro text-[10px] md:text-xs leading-relaxed transition-colors ${
            isOpen ? "text-(--ollie-cyan)" : "text-white/60 group-hover:text-white/80"
          }`}
        >
          {item.q}
        </span>
        <span className={`shrink-0 ml-4 transition-colors ${isOpen ? "text-(--ollie-cyan)" : "text-white/30"}`}>
          {isOpen ? <Minus size={14} /> : <Plus size={14} />}
        </span>
      </button>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5">
              <p className="text-white/50 text-sm leading-relaxed font-sans">{item.a}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// â”€â”€ Tab content components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function HelpFaqTab() {
  const [openIndex, setOpenIndex] = useState<number | null>(0)

  return (
    <div>
      <div className="text-center mb-10">
        <h2 className="retro text-xl md:text-2xl text-white mb-3">HELP CENTER</h2>
        <p className="retro text-(--ollie-cyan) text-[10px] tracking-widest">PRESS START TO GET ANSWERS</p>
      </div>
      <div className="max-w-2xl mx-auto flex flex-col gap-2">
        {FAQ_ITEMS.map((item, i) => (
          <AccordionItem
            key={i}
            item={item}
            index={i}
            isOpen={openIndex === i}
            onToggle={() => setOpenIndex(openIndex === i ? null : i)}
          />
        ))}
      </div>
    </div>
  )
}

function ChangelogTab() {
  return (
    <div>
      <div className="text-center mb-14">
        <h2 className="retro text-lg md:text-xl text-white mb-2">CHANGELOG</h2>
        <p className="retro text-[10px] text-white/30 tracking-widest">VERSION HISTORY</p>
      </div>
      <div className="relative max-w-2xl mx-auto">
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
                <div
                  className="absolute left-[12px] top-1 w-3 h-3 rounded-sm rotate-45 shrink-0"
                  style={{ background: accentColor, boxShadow: `0 0 10px ${accentColor}` }}
                />
                <div className="flex-1">
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
                  <h3 className="retro text-[11px] text-white mb-2 leading-loose">{entry.title}</h3>
                  <p className="text-white/40 text-sm leading-relaxed font-sans">{entry.desc}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function AboutTab() {
  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-12">
        <h2 className="retro text-lg md:text-xl text-white mb-3">ABOUT OLLIE</h2>
        <p className="retro text-[10px] text-white/30 tracking-widest">OPEN SOURCE FACE RECOGNITION</p>
      </div>

      <div className="flex flex-col gap-6">
        <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/10">
          <p className="text-white/70 text-sm leading-relaxed">
            Ollie is an open-source face recognition project built with PyTorch, trained on LFW and VGGFace2.
            It uses a custom Siamese ResNet backbone to learn face embeddings in a 256-dimensional space,
            then performs similarity search to match faces across a curated celebrity dataset.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Accuracy", value: "83.5%", sub: "LFW test set" },
            { label: "Identities", value: "9,131", sub: "in dataset" },
            { label: "Embeddings", value: "256-dim", sub: "L2 normalized" },
          ].map((stat) => (
            <div key={stat.label} className="p-4 rounded-xl bg-white/[0.03] border border-white/10 text-center">
              <div className="text-(--ollie-cyan) font-black text-xl mb-1">{stat.value}</div>
              <div className="text-white/60 text-xs font-semibold">{stat.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>

        <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/10">
          <h3 className="text-white font-bold text-sm mb-3">Stack</h3>
          <div className="flex flex-wrap gap-2">
            {["PyTorch", "FAISS", "InsightFace", "MediaPipe", "Next.js", "Three.js", "Framer Motion"].map((tech) => (
              <span key={tech} className="px-3 py-1 rounded-full border border-white/10 text-white/50 text-xs font-mono">
                {tech}
              </span>
            ))}
          </div>
        </div>

        <div className="flex justify-center">
          <Link
            href="/ai"
            className="group flex items-center gap-2 px-6 py-3 rounded-full bg-(--ollie-cyan)/10 border border-(--ollie-cyan)/30 text-(--ollie-cyan) text-sm font-semibold hover:bg-(--ollie-cyan)/20 transition-all"
          >
            Explore the Neural Architecture
            <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </div>
    </div>
  )
}

// â”€â”€ Main page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const TABS = [
  { id: "faq", label: "Help & FAQ" },
  { id: "changelog", label: "Changelog" },
  { id: "about", label: "About" },
] as const

type TabId = typeof TABS[number]["id"]

export default function InfoPage() {
  const [activeTab, setActiveTab] = useState<TabId>("faq")

  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="horizontal-lines" mask="fade-edges" fill="rgba(255,255,255,0.06)" size={28} className="fixed" />
      <Nav />

      <div className="pt-24 pb-32 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Tab bar */}
          <div className="flex justify-center mb-16">
            <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/10">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                    activeTab === tab.id
                      ? "text-black"
                      : "text-white/40 hover:text-white/70"
                  }`}
                >
                  {activeTab === tab.id && (
                    <motion.div
                      layoutId="tab-bg"
                      className="absolute inset-0 bg-(--ollie-cyan) rounded-lg"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                    />
                  )}
                  <span className="relative z-10">{tab.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Tab content */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.35, ease: "easeInOut" }}
            >
              {activeTab === "faq" && <HelpFaqTab />}
              {activeTab === "changelog" && <ChangelogTab />}
              {activeTab === "about" && <AboutTab />}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      <Footer />
    </main>
  )
}
