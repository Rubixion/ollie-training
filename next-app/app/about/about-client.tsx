"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, Minus, ArrowRight, Zap, Search, Star } from "lucide-react"
import Link from "next/link"
import RadialOrbitalTimeline from "@/components/ui/radial-orbital-timeline"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"

const FAQ_ITEMS = [
  {
    q: "How does the matching actually work?",
    a: "Your photo gets converted into a 256-number fingerprint by a custom neural network — a compact summary of your face's structure, proportions, and features. Ollie then finds the celebrities whose fingerprints are mathematically closest to yours. The whole process takes under 2 seconds.",
  },
  {
    q: "Is my photo stored or shared?",
    a: "No. Your image is processed in memory during the request and never saved to any server, database, or log. We don't collect, store, or share photos. Ever.",
  },
  {
    q: "How accurate are the results?",
    a: "The model reaches 83%+ accuracy on standard face recognition benchmarks. Real-world accuracy depends on photo quality — front-facing, evenly lit photos give the best matches. The model improves with every training run as more feedback comes in.",
  },
  {
    q: "My result looks wrong. What should I do?",
    a: "Use the Feedback page to flag it. Every correction gets fed directly into the next training run — your report makes the model more accurate for everyone, not just you.",
  },
  {
    q: "Does it cost anything?",
    a: "No. Ollie is completely free — no account required, no subscription, no ads.",
  },
]

const changelogData = [
  {
    id: 1,
    title: "v2.0",
    date: "Jun 2026",
    content: "Scaled training to 3.3M images using VGGFace2. InsightFace GPU-accelerated feature extraction. Age and gender weighting added for better match relevance. Biggest quality jump so far.",
    category: "Major Release",
    icon: Zap,
    relatedIds: [2],
    status: "completed" as const,
    energy: 100,
  },
  {
    id: 2,
    title: "v1.5",
    date: "Jun 2026",
    content: "Replaced brute-force similarity search with FAISS vector indexing, cutting search time by 100x. Threshold tuned for better precision. Three search modes added.",
    category: "Performance",
    icon: Search,
    relatedIds: [1, 3],
    status: "completed" as const,
    energy: 68,
  },
  {
    id: 3,
    title: "v1.0",
    date: "May 2026",
    content: "First working prototype. Siamese network trained on LFW dataset with MediaPipe facial geometry extraction. Slow but it worked.",
    category: "Initial Release",
    icon: Star,
    relatedIds: [2],
    status: "completed" as const,
    energy: 35,
  },
]

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
      transition={{ duration: 0.4, delay: index * 0.06 }}
      className="border-b border-white/8 last:border-0"
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between py-5 text-left group gap-6"
      >
        <span className={`text-sm font-semibold transition-colors leading-snug ${isOpen ? "text-white" : "text-white/60 group-hover:text-white/90"}`}>
          {item.q}
        </span>
        <span className={`shrink-0 transition-colors ${isOpen ? "text-white/60" : "text-white/20 group-hover:text-white/40"}`}>
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
            <p className="text-white/50 text-sm leading-relaxed pb-5">{item.a}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function AboutUsTab() {
  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-10">
        <h2 className="text-2xl font-black text-white mb-4 tracking-tight">Built from scratch, for everyone</h2>
        <div className="space-y-4 text-white/55 text-sm leading-relaxed">
          <p>
            Ollie started as a single question: could you train a neural network from scratch to match faces the
            way people do — not by memorising specific celebrities, but by learning what makes any face unique?
          </p>
          <p>
            The answer turned out to be yes. The model is now trained on 3.3 million faces across 9,131
            celebrity identities, and it keeps improving with every training run.
          </p>
          <p>
            Every match you get comes from that model running directly on your upload. No third-party APIs,
            no borrowed recognition services. Your photo is processed and immediately discarded — nothing is stored.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-10">
        {[
          { value: "83%+", label: "Accuracy", sub: "on benchmarks" },
          { value: "3.3M", label: "Training faces", sub: "VGGFace2 + LFW" },
          { value: "9,131", label: "Celebrities", sub: "in the dataset" },
          { value: "0", label: "Photos stored", sub: "ever" },
        ].map((stat) => (
          <div key={stat.label} className="p-4 rounded-2xl bg-white/[0.03] border border-white/8 text-center">
            <div className="text-2xl font-black text-white mb-1">{stat.value}</div>
            <div className="text-white/60 text-xs font-semibold">{stat.label}</div>
            <div className="text-white/25 text-xs mt-0.5">{stat.sub}</div>
          </div>
        ))}
      </div>

      <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8 mb-8">
        <p className="text-white/40 text-xs font-bold tracking-widest uppercase mb-4">Stack</p>
        <div className="flex flex-wrap gap-2">
          {["PyTorch", "FAISS", "InsightFace", "MediaPipe", "Next.js", "Three.js", "Framer Motion"].map((tech) => (
            <span key={tech} className="px-3 py-1 rounded-full border border-white/10 text-white/50 text-xs font-mono">
              {tech}
            </span>
          ))}
        </div>
      </div>

      <Link
        href="/ai"
        className="group inline-flex items-center gap-2 px-6 py-3 rounded-full bg-white/5 border border-white/10 text-white/70 text-sm font-semibold hover:bg-white/8 hover:text-white hover:border-white/20 transition-all"
      >
        How the AI works
        <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
      </Link>
    </div>
  )
}

function ChangelogTab() {
  return (
    <div>
      <div className="max-w-2xl mx-auto mb-4">
        <h2 className="text-2xl font-black text-white mb-1 tracking-tight">Changelog</h2>
        <p className="text-white/35 text-sm">Click a node to see what changed. Click background to reset.</p>
      </div>
      <RadialOrbitalTimeline
        timelineData={changelogData}
        className="h-[620px] rounded-2xl border border-white/8"
      />
    </div>
  )
}

function HelpFaqTab() {
  const [openIndex, setOpenIndex] = useState<number | null>(0)

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-10">
        <h2 className="text-2xl font-black text-white mb-2 tracking-tight">Common questions</h2>
        <p className="text-white/35 text-sm">Everything you might want to know before you upload.</p>
      </div>
      <div>
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
      <div className="mt-10 p-5 rounded-2xl bg-white/[0.02] border border-white/8">
        <p className="text-white/35 text-sm">
          Still have a question?{" "}
          <Link href="/contact" className="text-white/60 hover:text-white underline underline-offset-2 transition-colors">
            Get in touch
          </Link>
          {" "}and we&apos;ll get back to you within 48 hours.
        </p>
      </div>
    </div>
  )
}

const TABS = [
  { id: "about", label: "About Us" },
  { id: "changelog", label: "Changelog" },
  { id: "faq", label: "Help & FAQ" },
] as const

type TabId = typeof TABS[number]["id"]

export default function AboutPageClient() {
  const [activeTab, setActiveTab] = useState<TabId>("about")

  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="horizontal-lines" mask="fade-edges" fill="rgba(255,255,255,0.06)" size={28} className="fixed" />
      <Nav />

      <div className="pt-24 pb-32 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-center mb-14">
            <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/10">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                    activeTab === tab.id ? "text-black" : "text-white/40 hover:text-white/70"
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

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
            >
              {activeTab === "about" && <AboutUsTab />}
              {activeTab === "changelog" && <ChangelogTab />}
              {activeTab === "faq" && <HelpFaqTab />}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      <Footer />
    </main>
  )
}
