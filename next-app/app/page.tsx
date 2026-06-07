"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { ArrowRight, Upload, Search, Star, Brain, Globe, MessageSquare, CheckCircle, Zap, Users } from "lucide-react"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { ParticleTextEffect } from "@/components/effects/particle-text"

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block text-xs font-bold tracking-widest uppercase text-white/40 mb-4">
      {children}
    </span>
  )
}

function Divider() {
  return <div className="border-t border-white/5 max-w-6xl mx-auto" />
}

export default function Page() {
  return (
    <main className="relative bg-transparent overflow-hidden">
      <DottedSurface />
      <Nav />

      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="relative flex flex-col items-center justify-center min-h-screen pt-16 px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full max-w-4xl mx-auto"
        >
          <div className="w-full mb-4">
            <ParticleTextEffect words={["OLLIE", "FIND YOUR TWIN", "FACE AI", "WHO ARE YOU?"]} />
          </div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="text-white/40 text-sm md:text-base tracking-widest uppercase mb-10"
          >
            AI-powered celebrity face matching
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/find"
              className="group flex items-center gap-2 px-8 py-4 rounded-full bg-(--ollie-cyan) text-white font-bold text-sm hover:opacity-90 transition-all active:scale-[0.98]"
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
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.6 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        >
          <span className="text-white/20 text-xs tracking-widest uppercase">Scroll</span>
          <motion.div
            animate={{ y: [0, 6, 0] }}
            transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
            className="w-px h-8 bg-gradient-to-b from-white/20 to-transparent"
          />
        </motion.div>
      </section>

      {/* ── CELEBRITY FINDER ─────────────────────────────────────────────── */}
      <Divider />
      <section className="px-6 py-28 max-w-6xl mx-auto">
        <div className="flex flex-col lg:flex-row gap-16 items-center">
          {/* Left: text */}
          <motion.div
            initial={{ opacity: 0, x: -32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="flex-1"
          >
            <SectionLabel>Celebrity Finder</SectionLabel>
            <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight leading-tight mb-6">
              Which celebrity<br />do you look like?
            </h2>
            <p className="text-white/50 text-base leading-relaxed mb-8">
              Upload any photo and our AI compares your face against thousands of celebrities.
              It analyses your facial structure, bone geometry, and features to find who you most resemble - often with surprising accuracy.
            </p>

            {/* Steps */}
            <div className="flex flex-col gap-5 mb-10">
              {[
                { icon: Upload, step: "01", label: "Upload your photo", desc: "Drag & drop or browse. Works with any clear face photo." },
                { icon: Search, step: "02", label: "AI scans your face", desc: "256-dimensional face fingerprint created in under a second." },
                { icon: Star, step: "03", label: "See your matches", desc: "Top 5 celebrity matches ranked by similarity percentage." },
              ].map(({ icon: Icon, step, label, desc }) => (
                <div key={step} className="flex gap-4 items-start">
                  <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
                    <Icon size={18} className="text-white/50" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-white/20 font-mono text-xs">{step}</span>
                      <span className="text-white font-semibold text-sm">{label}</span>
                    </div>
                    <p className="text-white/40 text-sm">{desc}</p>
                  </div>
                </div>
              ))}
            </div>

            <Link
              href="/find"
              className="group inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-(--ollie-cyan) text-white font-bold text-sm hover:opacity-90 transition-all"
            >
              Try it now
              <ArrowRight size={15} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </motion.div>

          {/* Right: feature highlights */}
          <motion.div
            initial={{ opacity: 0, x: 32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="flex-1 grid grid-cols-2 gap-4"
          >
            {[
              { label: "3 search modes", desc: "CNN + Features, CNN only, or geometric features only" },
              { label: "Fast results", desc: "Your match appears in under 2 seconds" },
              { label: "No photo stored", desc: "Your image is never saved to any server" },
              { label: "Ranked by likeness", desc: "Each match shows a similarity percentage" },
              { label: "9,131 celebrities", desc: "Drawn from VGGFace2 and LFW datasets" },
              { label: "Improves over time", desc: "The AI learns from user feedback after each search" },
            ].map(({ label, desc }) => (
              <div key={label} className="p-5 rounded-2xl bg-white/[0.03] border border-white/8 flex flex-col gap-2">
                <CheckCircle size={15} className="text-sky-400" />
                <div className="text-white font-semibold text-sm">{label}</div>
                <div className="text-white/40 text-xs leading-relaxed">{desc}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── HOW THE AI WORKS ─────────────────────────────────────────────── */}
      <Divider />
      <section className="px-6 py-28 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <SectionLabel>The AI</SectionLabel>
          <h2 className="text-4xl md:text-5xl font-black text-white/80 tracking-tight mb-5">
            Custom-built face recognition
          </h2>
          <p className="text-white/45 text-base leading-relaxed max-w-2xl mx-auto">
            Ollie isn&apos;t using off-the-shelf software. It&apos;s a neural network trained from scratch on millions of face pairs -
            learning what makes two faces similar and what makes them different.
          </p>
        </motion.div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-14">
          {[
            { value: "83.5%", label: "Accuracy", sub: "on standard test set" },
            { value: "3.3M", label: "Training faces", sub: "VGGFace2 + LFW" },
            { value: "256", label: "Dimensions", sub: "in the face fingerprint" },
            { value: "32", label: "Face measurements", sub: "geometry + colour" },
          ].map((s) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="p-6 rounded-2xl bg-white/[0.03] border border-white/8 text-center"
            >
              <div className="text-3xl font-black text-white mb-1">{s.value}</div>
              <div className="text-white/60 text-sm font-semibold">{s.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{s.sub}</div>
            </motion.div>
          ))}
        </div>

        {/* Brief explanation columns */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {[
            {
              icon: Brain,
              title: "It learns like a brain",
              desc: "The network sees millions of face pairs - same person and different people - and slowly figures out what makes faces unique. No rules written by hand.",
            },
            {
              icon: Zap,
              title: "It measures and compares",
              desc: "Each face becomes a single number list (a 'fingerprint'). Comparing two faces is just measuring how far apart their fingerprints are.",
            },
            {
              icon: Users,
              title: "It gets smarter with use",
              desc: "Every time a user corrects a wrong match, that correction is added to the next round of training. The AI improves directly from real feedback.",
            },
          ].map(({ icon: Icon, title, desc }) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="p-6 rounded-2xl bg-white/[0.03] border border-white/8 flex flex-col gap-4"
            >
              <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                <Icon size={18} className="text-white/50" />
              </div>
              <div>
                <h3 className="text-white font-bold text-base mb-2">{title}</h3>
                <p className="text-white/40 text-sm leading-relaxed">{desc}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="text-center">
          <Link
            href="/neural"
            className="group inline-flex items-center gap-2 px-7 py-3.5 rounded-full border border-white/15 text-white/70 hover:text-white hover:border-white/30 font-semibold text-sm transition-all"
          >
            Deep dive into the architecture
            <ArrowRight size={15} className="group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </section>

      {/* ── FIND PERSON ──────────────────────────────────────────────────── */}
      <Divider />
      <section className="px-6 py-28 max-w-6xl mx-auto">
        <div className="flex flex-col lg:flex-row gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: 32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="flex-1 order-2 lg:order-1"
          >
            <div className="grid grid-cols-2 gap-3">
              {["Photo search", "Name lookup", "Social profiles", "News articles", "Public records", "Cross-platform"].map((f) => (
                <div key={f} className="px-4 py-3 rounded-xl bg-white/[0.03] border border-white/8 text-white/40 text-sm">
                  {f}
                </div>
              ))}
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: -32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="flex-1 order-1 lg:order-2"
          >
            <div className="flex items-center gap-3 mb-4">
              <SectionLabel>Find Person</SectionLabel>
              <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-white/10 text-white/40 border border-white/10 tracking-widest uppercase">Coming Soon</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight leading-tight mb-6">
              Find anyone,<br />anywhere online
            </h2>
            <p className="text-white/50 text-base leading-relaxed mb-6">
              The next step is internet-scale search. Upload a photo and find matching faces across public websites, social media, and news archives.
              This feature is in active development.
            </p>
            <p className="text-white/30 text-sm leading-relaxed">
              We&apos;re building this carefully - with privacy controls built in from the start, not added as an afterthought.
            </p>
          </motion.div>
        </div>
      </section>

      {/* ── FEEDBACK ─────────────────────────────────────────────────────── */}
      <Divider />
      <section className="px-6 py-28 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="flex flex-col md:flex-row items-center justify-between gap-10"
        >
          <div className="flex-1">
            <SectionLabel>Feedback</SectionLabel>
            <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight leading-tight mb-5">
              Help make<br />Ollie smarter
            </h2>
            <p className="text-white/50 text-base leading-relaxed mb-4">
              Every correction you submit - marking a match as wrong or right - gets fed directly into the next training run.
              Your feedback isn&apos;t just a rating. It becomes real training data that changes how the AI behaves.
            </p>
            <p className="text-white/30 text-sm leading-relaxed">
              This is collaborative AI training: the more people use and correct it, the better it gets for everyone.
            </p>
          </div>
          <div className="flex-1 flex flex-col gap-4">
            <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="flex items-center gap-3 mb-3">
                <MessageSquare size={18} className="text-white/40" />
                <span className="text-white font-semibold text-sm">Got a match wrong?</span>
              </div>
              <p className="text-white/40 text-sm leading-relaxed mb-5">
                Tell us it&apos;s wrong and the model will never make the same mistake on that face pair again.
              </p>
              <Link
                href="/feedback"
                className="group inline-flex items-center gap-2 px-6 py-3 rounded-full bg-(--ollie-cyan) text-white font-bold text-sm hover:opacity-90 transition-all"
              >
                Submit feedback
                <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
            <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
              <Globe size={18} className="text-white/40 mb-3" />
              <span className="text-white font-semibold text-sm block mb-2">Open model</span>
              <p className="text-white/40 text-sm leading-relaxed">
                The weights, training code, and architecture are all open. Build on it, improve it, fork it.
              </p>
            </div>
          </div>
        </motion.div>
      </section>

      <Footer />
    </main>
  )
}
