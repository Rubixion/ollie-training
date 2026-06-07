"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { ParticleTextEffect } from "@/components/effects/particle-text"
import { FaceFeatureViz } from "@/components/effects/face-feature-viz"
import { NeuralDeepViz } from "@/components/neural-deep-viz"

function SectionNum({ n }: { n: string }) {
  return <span className="text-(--ollie-cyan) font-mono text-xs tracking-widest font-bold block mb-1">{n}</span>
}

function Callout({ children }: { children: React.ReactNode }) {
  return (
    <div className="border-l-2 border-(--ollie-cyan)/40 pl-5 my-6 text-white/50 text-sm leading-relaxed italic">
      {children}
    </div>
  )
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block px-2 py-0.5 rounded bg-white/8 border border-white/10 text-white/60 text-xs font-mono mx-0.5">
      {children}
    </span>
  )
}

function FeatureRow({ index, name, plain }: { index: string; name: string; plain: string }) {
  return (
    <div className="flex gap-4 py-2.5 border-b border-white/5 last:border-0 text-xs">
      <span className="text-(--ollie-cyan) font-mono w-7 shrink-0">{index}</span>
      <span className="text-white/70 font-mono w-36 shrink-0">{name}</span>
      <span className="text-white/40">{plain}</span>
    </div>
  )
}

function Section({
  id,
  num,
  title,
  children,
}: {
  id: string
  num: string
  title: string
  children: React.ReactNode
}) {
  return (
    <motion.div
      id={id}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.7 }}
      className="py-24 px-6 max-w-5xl mx-auto border-t border-white/5 scroll-mt-20"
    >
      <SectionNum n={num} />
      <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight mb-10">{title}</h2>
      {children}
    </motion.div>
  )
}

function ArchitectureScroll() {
  return (
    <div className="mt-8">

      {/* ── 01  What is a neural network? ──────────────────────────────── */}
      <Section id="s-backbone" num="01" title="What even is a neural network?">
        <div className="flex flex-col lg:flex-row gap-12 items-start">
          <div className="flex-1">
            <p className="text-white/60 text-base leading-relaxed mb-6">
              A neural network is software that learns by looking at examples - a lot of examples.
              You don&apos;t write rules for it. You show it thousands of pictures of cats and dogs and it figures out,
              on its own, what makes a cat look like a cat.
            </p>
            <p className="text-white/60 text-base leading-relaxed mb-6">
              Inside, it&apos;s made of layers. Each layer looks at the previous layer&apos;s output and finds patterns in it.
              Early layers notice edges and colours. Middle layers notice shapes like eyes and noses.
              Deep layers notice whole faces and people.
            </p>
            <Callout>
              Think of it like a chain of workers on a production line. Worker 1 looks for edges.
              Worker 2 looks for shapes made of those edges. Worker 3 looks for faces made of those shapes.
              No worker needed to be taught what a face is - they each learned their job just from seeing examples.
            </Callout>
            <p className="text-white/60 text-base leading-relaxed">
              Ollie&apos;s network has 8 of these layers, called <strong className="text-white/85">residual blocks</strong>.
              The &quot;residual&quot; part means each block also passes its raw input straight through to the next block -
              this stops the network from forgetting what it learned in earlier layers as it gets deeper.
            </p>
          </div>
          <div className="shrink-0 flex flex-col items-center gap-4">
            <div className="rounded-2xl overflow-hidden border border-white/10 bg-transparent/40 p-2">
              <FaceFeatureViz size={280} />
            </div>
            <p className="text-white/30 text-xs text-center max-w-[260px]">
              The network maps your face to 68 landmark points, then measures the geometry between them
            </p>
          </div>
        </div>
      </Section>

      {/* ── 02  Why CNN? ────────────────────────────────────────────────── */}
      <Section id="s-siamese" num="02" title="Why does it need to see faces specifically?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          A standard neural network trained on random images - cars, chairs, animals - is useless for faces.
          It doesn&apos;t know that the distance between your eyes matters, or that your jaw shape is distinctive.
          Ollie&apos;s network was trained from scratch using only face pairs - it never saw anything else.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          The type of network used here is called a <strong className="text-white/85">CNN</strong> - Convolutional Neural Network.
          Instead of looking at every pixel at once, it slides a small window across the image and looks for patterns
          in small regions. This is exactly how we notice that someone has a strong jaw or wide-set eyes.
        </p>

        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {[
            { step: "Input", desc: "Your photo - just pixels, 96×96" },
            { step: "Early layers", desc: "Find edges, colour changes, basic shapes" },
            { step: "Deep layers", desc: "Find eyes, noses, jaw shapes" },
          ].map(({ step, desc }) => (
            <div key={step} className="p-5 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="text-white font-bold text-sm mb-2">{step}</div>
              <div className="text-white/40 text-sm leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>

        <p className="text-white/60 text-base leading-relaxed mb-6">
          At the end, the network squashes your whole face down into a list of 256 numbers.
          This list is called an <strong className="text-white/85">embedding</strong> or &quot;face fingerprint.&quot;
          Every face gets one. Faces that look similar end up with fingerprints that are mathematically close together.
        </p>
        <Callout>
          The number 256 isn&apos;t special - it&apos;s just enough dimensions to capture the complexity of a face
          without wasting memory. More dimensions = more detail, but also more to compute and more risk of memorising
          training data instead of learning general patterns.
        </Callout>
      </Section>

      {/* ── 03  Siamese (how it learns similarity) ──────────────────────── */}
      <Section id="s-contrastive" num="03" title="How does it learn what 'similar' means?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Learning to recognise faces is different from learning to label objects.
          The network doesn&apos;t need to say &quot;that&apos;s Brad Pitt.&quot; It needs to say &quot;those two photos are the same person&quot;
          or &quot;those two photos are different people.&quot; That&apos;s a harder and more useful skill.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          To do this, Ollie uses a <strong className="text-white/85">Siamese network</strong>.
          Imagine twins sitting side by side - each one looking at a different photo, using the exact same brain.
          They each produce a fingerprint. Then a judge looks at how different those fingerprints are.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <div className="text-white font-bold text-sm mb-3">Same person → fingerprints should be close</div>
            <p className="text-white/40 text-sm leading-relaxed">
              Two photos of you, five years apart. Different lighting, different angle.
              The network learns that the distance between your two fingerprints should be nearly zero.
            </p>
          </div>
          <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <div className="text-white font-bold text-sm mb-3">Different people → fingerprints should be far apart</div>
            <p className="text-white/40 text-sm leading-relaxed">
              A photo of you and a photo of a stranger. The network learns that the distance between your fingerprints
              should be at least 2.0 units in the fingerprint space.
            </p>
          </div>
        </div>

        <p className="text-white/60 text-base leading-relaxed mb-6">
          The thing that teaches the network to do this is called the <strong className="text-white/85">loss function</strong>.
          Think of it as a score that tells the network how badly it did. If it put two same-person fingerprints far apart,
          or two different-person fingerprints close together - it gets penalised.
          Over millions of tries, it learns to stop making those mistakes.
        </p>
        <Callout>
          The network sees 3.3 million face photos during training. For each pair, it checks its answer,
          calculates the penalty, and adjusts itself slightly in the right direction.
          After enough adjustments, it&apos;s learned the shape of what &quot;same person&quot; looks like in fingerprint space.
        </Callout>
      </Section>

      {/* ── 04  Geometric measurements ──────────────────────────────────── */}
      <Section id="s-features" num="04" title="What does it actually measure on your face?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          On top of the 256-number fingerprint from the neural network, Ollie also takes 32 direct physical
          measurements from your face. These are things a doctor or forensic artist might measure - not things
          a neural network invented.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          It finds 478 specific points on your face (like corners of your eyes, tip of your nose, edges of your lips)
          and then calculates ratios and distances between them. Ratios work better than raw measurements because they
          don&apos;t depend on how big your photo is.
        </p>

        <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden mb-6">
          <div className="px-5 py-3 border-b border-white/5 flex gap-4 text-xs text-white/25 font-mono">
            <span className="w-7">#</span>
            <span className="w-36">name</span>
            <span>what it means in plain English</span>
          </div>
          <div className="px-5 pb-2">
            <FeatureRow index="0" name="eye_dist" plain="How far apart your eyes are, relative to your face width" />
            <FeatureRow index="1–2" name="ear_L / ear_R" plain="How open each eye is (eye aspect ratio)" />
            <FeatureRow index="3–9" name="iris_*" plain="Colour and texture of each iris - hue, saturation, variance" />
            <FeatureRow index="9" name="face_ratio" plain="Face height ÷ face width - are you more oval or round?" />
            <FeatureRow index="10" name="jaw_w" plain="How wide your jaw is relative to your face" />
            <FeatureRow index="12–13" name="nose_w / nose_h" plain="Width and height of your nose relative to your face" />
            <FeatureRow index="14–16" name="lip_*" plain="Width, height, and openness of your mouth" />
            <FeatureRow index="17–19" name="hair_H/S/V" plain="Colour of your hair - hue, saturation, brightness" />
            <FeatureRow index="20–22" name="skin_L/a/b" plain="Skin tone in a colour system that matches how humans perceive colour" />
            <FeatureRow index="23–26" name="*_ratio" plain="Proportions between forehead, eyes, nose, mouth, and chin" />
            <FeatureRow index="28" name="age_norm" plain="Estimated age (0–100 scaled to 0–1)" />
            <FeatureRow index="29" name="gender_score" plain="Predicted gender from facial structure (1 = male, 0 = female)" />
          </div>
        </div>

        <p className="text-white/60 text-base leading-relaxed">
          These 32 measurements are used to double-check the neural network&apos;s top results.
          If the network thinks you match someone but they&apos;re 30 years older or a different gender,
          the geometric check will push that result down the list.
        </p>
      </Section>

      {/* ── 05  FAISS search ───────────────────────────────────────────── */}
      <Section id="s-faiss" num="05" title="How does it find your match so fast?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Finding the closest fingerprint out of 17,000+ celebrities could take a long time if you checked every single one.
          Ollie uses a tool called <strong className="text-white/85">FAISS</strong> (built by Meta) to do this search
          almost instantly - in about 10 milliseconds.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          FAISS stores all the celebrity fingerprints in a way that makes comparison very efficient.
          When you upload your photo, your fingerprint is created and then compared against every stored fingerprint.
          The 200 closest ones come back instantly.
        </p>

        <div className="flex flex-col gap-3 mb-8">
          {[
            { step: "1", label: "You upload a photo", desc: "The network converts it into your 256-number fingerprint." },
            { step: "2", label: "FAISS searches all celebrities", desc: "Finds the 200 fingerprints most similar to yours. Takes ~10ms." },
            { step: "3", label: "Geometric re-ranking", desc: "The 32 physical measurements are checked. Bad matches on age, gender, or skin tone are pushed down." },
            { step: "4", label: "Top 5 shown to you", desc: "The closest matches after both checks, ranked by similarity percentage." },
          ].map(({ step, label, desc }) => (
            <div key={step} className="flex gap-4 p-5 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="w-8 h-8 rounded-full border border-white/15 flex items-center justify-center text-white/40 font-mono text-xs shrink-0">
                {step}
              </div>
              <div>
                <div className="text-white font-semibold text-sm mb-1">{label}</div>
                <div className="text-white/40 text-sm">{desc}</div>
              </div>
            </div>
          ))}
        </div>

        <Callout>
          The similarity percentage you see isn&apos;t a raw score - it&apos;s been converted from the fingerprint distance
          into something readable. A 95% match means the fingerprints are very close together in that 256-dimensional space.
          A 60% match means they&apos;re noticeable further apart, but still in the same neighbourhood.
        </Callout>
      </Section>

      {/* ── 06  Feedback loop ──────────────────────────────────────────── */}
      <Section id="s-feedback" num="06" title="How does it get better over time?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Every AI system makes mistakes. What makes Ollie different is that it learns from its mistakes in real time -
          because you can tell it when it got something wrong.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          When you mark a match as wrong, that specific face pair - your photo and the wrong celebrity - gets stored
          as a &quot;this should be far apart&quot; example. When Ollie is retrained, it sees that example and adjusts
          so it never makes the same mistake on a face like that again.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <div className="text-white font-bold text-sm mb-3">What happens when you submit feedback</div>
            <ol className="text-white/40 text-sm leading-relaxed list-decimal list-inside space-y-2">
              <li>Your face + the wrong match are saved as a pair</li>
              <li>It&apos;s labelled &quot;different person&quot;</li>
              <li>Next training run includes this pair</li>
              <li>The network learns to push them apart</li>
              <li>Everyone gets better results from then on</li>
            </ol>
          </div>
          <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <div className="text-white font-bold text-sm mb-3">This isn&apos;t just a rating system</div>
            <p className="text-white/40 text-sm leading-relaxed">
              Most apps collect feedback and do nothing with it. Here, your correction is treated exactly the same
              as a labelled training example from a professional dataset. It goes directly into the next training run -
              no human review, no filtering.
            </p>
          </div>
        </div>

        <div className="flex">
          <Link
            href="/feedback"
            className="group flex items-center gap-2 px-6 py-3 rounded-full bg-(--ollie-cyan)/10 border border-(--ollie-cyan)/30 text-(--ollie-cyan) text-sm font-semibold hover:bg-(--ollie-cyan)/20 transition-all"
          >
            Submit a correction
            <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </Section>

      {/* ── 07  Training ───────────────────────────────────────────────── */}
      <Section id="s-training" num="07" title="How long did it take to train?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Training a neural network isn&apos;t like installing software. It&apos;s a process that runs for hours or days,
          showing the network millions of examples and gradually improving its answers each time.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Ollie was trained on a consumer gaming GPU - a GTX 1650 with 4GB of memory. Not a supercomputer.
          The best result so far is <strong className="text-white/85">83.5% accuracy</strong> on a standard face-matching test,
          reached after 90 rounds of training. It&apos;s still improving.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          {[
            { label: "Best accuracy", value: "83.5%", sub: "on LFW test pairs" },
            { label: "Celebrities in index", value: "9,131", sub: "VGGFace2 + LFW" },
            { label: "Photos trained on", value: "3.3M", sub: "face pair samples" },
            { label: "Fingerprint size", value: "256", sub: "numbers per face" },
          ].map((stat) => (
            <div key={stat.label} className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 text-center">
              <div className="text-(--ollie-cyan) font-black text-2xl mb-1">{stat.value}</div>
              <div className="text-white/70 text-xs font-semibold">{stat.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>

        <p className="text-white/50 text-sm leading-relaxed">
          Each round of training is called an <Pill>epoch</Pill>. In each epoch, the network sees every training photo once,
          checks how wrong its answers were, and nudges itself in the right direction.
          The nudge size (called <Pill>learning rate</Pill>) starts at 0.0001 and shrinks over time so the network
          stops jumping around and settles into the best version of itself.
        </p>
      </Section>
    </div>
  )
}

export default function NeuralPage() {
  return (
    <main className="relative bg-transparent text-white">
      <DottedSurface />
      <Nav />

      {/* Hero */}
      <section className="relative min-h-[40vh] flex flex-col items-center justify-center pt-24 pb-12 px-6 text-center">
        <div className="w-full max-w-7xl mx-auto mb-6">
          <ParticleTextEffect
            words={["HOW IT WORKS", "THE AI EXPLAINED", "FACIAL FINGERPRINTS", "NEURAL NETWORKS", "PIXELS TO MATCH"]}
          />
        </div>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="text-white/40 text-xs tracking-widest uppercase"
        >
          No jargon. Start from zero.
        </motion.p>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="text-white/25 text-xs mt-2"
        >
          Hover or click any node in the 3D scene below to explore each layer
        </motion.p>
      </section>

      {/* 3D Interactive network */}
      <section className="px-6 pb-16 max-w-6xl mx-auto">
        <NeuralDeepViz />
      </section>

      {/* Architecture explanations */}
      <ArchitectureScroll />

      <Footer />
    </main>
  )
}
