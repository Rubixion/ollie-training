"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { Camera, Brain, Globe, MessageSquare, ArrowRight } from "lucide-react"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { ParticleTextEffect } from "@/components/effects/particle-text"

const FEATURE_CARDS = [
  {
    icon: Camera,
    title: "Celebrity Finder",
    desc: "Upload your face. Find your celebrity twin.",
    href: "/find",
    badge: null,
    color: "cyan",
  },
  {
    icon: Brain,
    title: "How It Works",
    desc: "Deep dive into the CNN architecture powering Ollie.",
    href: "/neural",
    badge: null,
    color: "purple",
  },
  {
    icon: Globe,
    title: "Find Person",
    desc: "Search anyone online. Coming soon.",
    href: "/find-person",
    badge: "BETA",
    color: "white",
  },
  {
    icon: MessageSquare,
    title: "Feedback",
    desc: "Help train the model. Your corrections matter.",
    href: "/feedback",
    badge: null,
    color: "cyan",
  },
]

export default function Page() {
  return (
    <main className="relative min-h-screen bg-black overflow-hidden">
      <DottedSurface />
      <Nav />

      {/* Hero */}
      <section className="relative flex flex-col items-center justify-center min-h-screen pt-16 px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full max-w-4xl mx-auto"
        >
          {/* Particle text title */}
          <div className="w-full mb-4">
            <ParticleTextEffect words={["OLLIE", "FIND YOUR TWIN", "FACE AI", "WHO ARE YOU?"]} />
          </div>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="text-white/40 text-sm md:text-base tracking-widest uppercase mb-10"
          >
            AI-powered celebrity face matching
          </motion.p>

          {/* CTA buttons */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/find"
              className="group flex items-center gap-2 px-8 py-4 rounded-full bg-[--ollie-cyan] text-black font-bold text-sm hover:opacity-90 transition-all shadow-lg shadow-[--ollie-glow] active:scale-[0.98]"
            >
              Find Your Lookalike
              <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/neural"
              className="flex items-center gap-2 px-8 py-4 rounded-full border border-white/15 text-white/70 hover:text-white hover:border-white/30 font-semibold text-sm transition-all"
            >
              See How It Works
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll hint */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.6 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        >
          <span className="text-white/20 text-xs tracking-widest uppercase">Explore</span>
          <motion.div
            animate={{ y: [0, 6, 0] }}
            transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
            className="w-px h-8 bg-gradient-to-b from-white/20 to-transparent"
          />
        </motion.div>
      </section>

      {/* Feature cards */}
      <section className="relative px-6 pb-32 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-white/30 text-xs tracking-widest uppercase font-semibold">What Ollie can do</span>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURE_CARDS.map((card, i) => {
            const Icon = card.icon
            const accentClass =
              card.color === "cyan"
                ? "text-[--ollie-cyan]"
                : card.color === "purple"
                  ? "text-[--ollie-purple]"
                  : "text-white/60"
            const glowClass =
              card.color === "cyan"
                ? "hover:shadow-[0_0_30px_var(--ollie-glow)]"
                : card.color === "purple"
                  ? "hover:shadow-[0_0_30px_oklch(0.65_0.22_290_/_15%)]"
                  : "hover:shadow-[0_0_30px_rgba(255,255,255,0.05)]"

            return (
              <motion.div
                key={card.href}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
              >
                <Link
                  href={card.href}
                  className={`group relative flex flex-col gap-4 p-6 rounded-2xl bg-white/5 backdrop-blur border border-white/10 hover:border-white/20 hover:bg-white/[0.08] transition-all duration-300 ${glowClass} h-full`}
                >
                  {/* Badge */}
                  {card.badge && (
                    <span className="absolute top-4 right-4 text-[9px] font-bold px-2 py-0.5 rounded bg-[--ollie-purple]/20 text-[--ollie-purple] border border-[--ollie-purple]/30 tracking-widest">
                      {card.badge}
                    </span>
                  )}

                  {/* Icon */}
                  <div className={`w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center ${accentClass} group-hover:scale-110 transition-transform`}>
                    <Icon size={20} />
                  </div>

                  {/* Text */}
                  <div>
                    <h3 className="text-white font-bold text-base mb-1">{card.title}</h3>
                    <p className="text-white/40 text-sm leading-relaxed">{card.desc}</p>
                  </div>

                  {/* Arrow */}
                  <div className={`mt-auto flex items-center gap-1 text-xs font-semibold ${accentClass} opacity-0 group-hover:opacity-100 transition-opacity`}>
                    Open <ArrowRight size={12} />
                  </div>
                </Link>
              </motion.div>
            )
          })}
        </div>
      </section>

      <Footer />
    </main>
  )
}
