"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, Minus } from "lucide-react"

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
    a: "Not yet. Internet search is in development under 'Find Person' — check back soon.",
  },
  {
    q: "HOW DO I HELP IMPROVE IT?",
    a: "Use the Feedback section to correct bad matches. Every submission makes the model smarter.",
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
      transition={{ duration: 0.4, delay: index * 0.07 }}
      className="border border-white/8 rounded-none"
      style={{
        borderLeft: "3px solid",
        borderLeftColor: isOpen ? "var(--ollie-cyan)" : "oklch(1 0 0 / 8%)",
        background: isOpen ? "oklch(1 0 0 / 2%)" : "transparent",
        transition: "border-left-color 0.2s, background 0.2s",
      }}
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-5 py-4 text-left group"
      >
        <span
          className={`retro text-[10px] md:text-xs leading-relaxed transition-colors ${
            isOpen ? "text-[--ollie-cyan]" : "text-white/60 group-hover:text-white/80"
          }`}
        >
          {item.q}
        </span>
        <span className={`shrink-0 ml-4 transition-colors ${isOpen ? "text-[--ollie-cyan]" : "text-white/30"}`}>
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

export function FaqSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(0)

  return (
    <section id="faq" className="py-24 md:py-32 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          {/* Pixel character */}
          <div className="text-5xl mb-6 select-none" role="img" aria-label="gamepad">
            🎮
          </div>

          <h2 className="retro text-xl md:text-2xl text-white mb-3">HELP CENTER</h2>
          <p className="retro text-[--ollie-cyan] text-[10px] tracking-widest">
            PRESS START TO GET ANSWERS
          </p>
        </motion.div>

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
    </section>
  )
}
