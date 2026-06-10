"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { DottedSurface } from "@/components/ui/dotted-surface"
import { ParticleTextEffect } from "@/components/effects/particle-text"
import { NeuralDeepViz } from "@/components/neural-deep-viz"

// ── helpers ──────────────────────────────────────────────────────────────────

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

// ── visualisations ────────────────────────────────────────────────────────────

function LayerDiagram() {
  const input  = [35, 58, 81].map((y) => ({ x: 36, y }))
  const h1     = [22, 42, 58, 74, 94].map((y) => ({ x: 118, y }))
  const h2     = [22, 42, 58, 74, 94].map((y) => ({ x: 200, y }))
  const output = [{ x: 282, y: 58 }]

  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-5 overflow-hidden">
      <div className="text-white/30 text-xs font-mono mb-3 tracking-widest text-center">FORWARD PASS</div>
      <svg viewBox="0 0 320 118" className="w-full">
        {/* connections input→h1 */}
        {input.flatMap((a, i) =>
          h1.map((b, j) => (
            <line key={`ih1-${i}-${j}`} x1={a.x + 7} y1={a.y} x2={b.x - 7} y2={b.y}
              stroke="rgba(100,200,255,0.07)" strokeWidth="0.7" />
          ))
        )}
        {/* connections h1→h2 */}
        {h1.flatMap((a, i) =>
          h2.map((b, j) => (
            <line key={`h1h2-${i}-${j}`} x1={a.x + 7} y1={a.y} x2={b.x - 7} y2={b.y}
              stroke="rgba(150,140,255,0.07)" strokeWidth="0.7" />
          ))
        )}
        {/* connections h2→output */}
        {h2.map((a, i) => (
          <line key={`h2o-${i}`} x1={a.x + 7} y1={a.y} x2={output[0].x - 9} y2={output[0].y}
            stroke="rgba(100,200,255,0.12)" strokeWidth="0.7" />
        ))}
        {/* input nodes */}
        {input.map((n, i) => (
          <circle key={`i-${i}`} cx={n.x} cy={n.y} r={7}
            fill="rgba(100,200,255,0.12)" stroke="rgba(100,200,255,0.55)" strokeWidth="1" />
        ))}
        {/* h1 nodes */}
        {h1.map((n, i) => (
          <circle key={`h1-${i}`} cx={n.x} cy={n.y} r={7}
            fill="rgba(140,140,255,0.10)" stroke="rgba(140,140,255,0.40)" strokeWidth="1" />
        ))}
        {/* h2 nodes */}
        {h2.map((n, i) => (
          <circle key={`h2-${i}`} cx={n.x} cy={n.y} r={7}
            fill="rgba(140,140,255,0.10)" stroke="rgba(140,140,255,0.40)" strokeWidth="1" />
        ))}
        {/* output node */}
        <circle cx={output[0].x} cy={output[0].y} r={9}
          fill="rgba(100,200,255,0.18)" stroke="rgba(100,200,255,0.9)" strokeWidth="1.5" />
        {/* labels */}
        <text x={36}  y={113} textAnchor="middle" fill="rgba(255,255,255,0.18)" fontSize="6.5" fontFamily="monospace">INPUTS</text>
        <text x={118} y={113} textAnchor="middle" fill="rgba(255,255,255,0.18)" fontSize="6.5" fontFamily="monospace">LAYER 1</text>
        <text x={200} y={113} textAnchor="middle" fill="rgba(255,255,255,0.18)" fontSize="6.5" fontFamily="monospace">LAYER 2</text>
        <text x={282} y={113} textAnchor="middle" fill="rgba(255,255,255,0.18)" fontSize="6.5" fontFamily="monospace">OUTPUT</text>
      </svg>
      <p className="text-white/25 text-xs text-center mt-2">
        Every node multiplies its inputs by learned weights, sums them, and passes the result forward.
      </p>
    </div>
  )
}

function EmbeddingSpaceViz() {
  const clusters = [
    { cx: 72,  cy: 70,  stroke: "rgba(100,210,255,0.9)", fill: "rgba(100,210,255,0.18)", dim: "rgba(100,210,255,0.15)", label: "Person A" },
    { cx: 200, cy: 58,  stroke: "rgba(180,130,255,0.9)", fill: "rgba(180,130,255,0.18)", dim: "rgba(180,130,255,0.15)", label: "Person B" },
    { cx: 62,  cy: 178, stroke: "rgba(100,255,175,0.9)", fill: "rgba(100,255,175,0.18)", dim: "rgba(100,255,175,0.15)", label: "Person C" },
    { cx: 210, cy: 185, stroke: "rgba(255,185,100,0.9)", fill: "rgba(255,185,100,0.18)", dim: "rgba(255,185,100,0.15)", label: "Person D" },
  ]
  const offsets = [[-14, -8], [12, -14], [16, 8], [-6, 16], [2, 2]]

  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-5 overflow-hidden">
      <div className="text-white/30 text-xs font-mono mb-3 tracking-widest text-center">512-D EMBEDDING SPACE (2D SLICE)</div>
      <svg viewBox="0 0 280 238" className="w-full">
        {/* grid */}
        {[0,1,2,3].map(i => (
          <line key={`gh-${i}`} x1={12} y1={12 + i * 65} x2={268} y2={12 + i * 65}
            stroke="rgba(255,255,255,0.035)" strokeWidth="0.5" />
        ))}
        {[0,1,2,3].map(i => (
          <line key={`gv-${i}`} x1={12 + i * 82} y1={12} x2={12 + i * 82} y2={220}
            stroke="rgba(255,255,255,0.035)" strokeWidth="0.5" />
        ))}
        {/* intra-cluster lines */}
        {clusters.map(cl =>
          offsets.flatMap(([ox, oy], j) =>
            offsets.slice(j + 1).map(([ox2, oy2], k) => (
              <line key={`cl-${cl.label}-${j}-${k}`}
                x1={cl.cx + ox} y1={cl.cy + oy}
                x2={cl.cx + ox2} y2={cl.cy + oy2}
                stroke={cl.dim} strokeWidth="0.9" />
            ))
          )
        )}
        {/* dots */}
        {clusters.map(cl =>
          offsets.map(([ox, oy], j) => (
            <circle key={`d-${cl.label}-${j}`}
              cx={cl.cx + ox} cy={cl.cy + oy} r={4.5}
              fill={cl.fill} stroke={cl.stroke} strokeWidth="1.2" />
          ))
        )}
        {/* cluster labels */}
        {clusters.map(cl => (
          <text key={`lbl-${cl.label}`} x={cl.cx} y={cl.cy + 30}
            textAnchor="middle" fill={cl.stroke} fontSize="6.5" fontFamily="monospace" opacity="0.75">
            {cl.label}
          </text>
        ))}
        {/* axis labels */}
        <text x={140} y={234} textAnchor="middle" fill="rgba(255,255,255,0.18)" fontSize="6" fontFamily="monospace">DIMENSION 1 OF 512</text>
      </svg>
      <p className="text-white/25 text-xs text-center mt-2">
        Each dot is one face photo. Same person → clustered together. Different person → far apart.
      </p>
    </div>
  )
}

// ── page sections ─────────────────────────────────────────────────────────────

function ArchitectureScroll() {
  return (
    <div className="mt-8">

      {/* 01 ── What is a neuron? */}
      <Section id="s-neuron" num="01" title="What even is a neural network?">
        <div className="flex flex-col lg:flex-row gap-12 items-start">
          <div className="flex-1">
            <p className="text-white/60 text-base leading-relaxed mb-6">
              Your brain contains about 86 billion neurons. Each one does something simple: it collects signals from
              other neurons, adds them up, and if the total is big enough, it fires and passes a signal on.
              That&apos;s it. The entire complexity of human thought is just billions of those simple operations happening in parallel.
            </p>
            <p className="text-white/60 text-base leading-relaxed mb-6">
              An artificial neuron works the same way. It takes a list of numbers as input, multiplies each one by a
              <strong className="text-white/85"> weight</strong> (a number that says how important that input is),
              sums everything up, and runs the result through an <strong className="text-white/85">activation function</strong> that
              decides how strongly to fire. The weights are what get learned — they start random and slowly get corrected.
            </p>
            <Callout>
              A single neuron can learn one pattern. Thousands of neurons in parallel can learn a whole layer of patterns.
              Stack enough layers and you get something that can learn anything — including faces.
            </Callout>
            <p className="text-white/60 text-base leading-relaxed">
              Each layer feeds into the next. Early layers learn simple things (edges, brightness changes).
              Middle layers combine those into shapes (eyes, jawlines). Deep layers recognise entire faces and identities.
              No layer was told what to look for — they all figured it out from examples.
            </p>
          </div>
          <div className="shrink-0 w-full lg:w-[320px] flex flex-col gap-4">
            <LayerDiagram />
            <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-4 text-xs text-white/40 space-y-2 font-mono">
              <div className="text-white/25 text-[10px] tracking-widest mb-2">SINGLE NEURON MATH</div>
              <div>output = activate( Σ(xᵢ × wᵢ) + b )</div>
              <div className="text-white/25">x = inputs&nbsp;&nbsp; w = weights&nbsp;&nbsp; b = bias</div>
            </div>
          </div>
        </div>
      </Section>

      {/* 02 ── How does it learn? */}
      <Section id="s-learning" num="02" title="How does it actually learn?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          A fresh network has random weights and gives wrong answers. Learning is the process of making the weights less wrong.
          It happens in four steps that repeat millions of times:
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-8">
          {[
            {
              step: "1. Forward pass",
              desc: "Feed an image in. Let it flow through every layer. Get an answer out the other end.",
            },
            {
              step: "2. Compute the loss",
              desc: "Compare the answer to the correct answer. The loss function returns a single number saying how wrong it was. Lower is better.",
            },
            {
              step: "3. Backpropagation",
              desc: "Starting from the loss, work backwards through every layer. Calculate how much each weight contributed to the error.",
            },
            {
              step: "4. Update weights",
              desc: "Nudge every weight slightly in the direction that reduces the loss. The size of the nudge is called the learning rate.",
            },
          ].map(({ step, desc }) => (
            <div key={step} className="p-5 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="text-white font-bold text-sm mb-2">{step}</div>
              <div className="text-white/40 text-sm leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>

        <p className="text-white/60 text-base leading-relaxed mb-6">
          One full pass over all training data is called an <Pill>epoch</Pill>.
          Ollie trains for 35 epochs over 5.8 million photos — that&apos;s about 400 million individual forward passes.
          The <Pill>learning rate</Pill> starts at 0.1 and drops by 10× at epochs 10, 20, and 25,
          so the network makes big corrections early and tiny fine-tuning corrections at the end.
        </p>

        <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-6 mb-6">
          <div className="text-white/40 text-xs font-mono tracking-widest mb-4">RLHF vs RLVR vs WHAT OLLIE USES</div>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <div className="text-white/80 font-semibold text-sm mb-2">RLHF</div>
              <p className="text-white/40 text-xs leading-relaxed">
                Used in ChatGPT and Claude. Human raters score outputs. A separate "reward model" learns those preferences.
                The main model is fine-tuned to score higher. Requires thousands of human hours.
              </p>
            </div>
            <div>
              <div className="text-white/80 font-semibold text-sm mb-2">RLVR</div>
              <p className="text-white/40 text-xs leading-relaxed">
                Used in math and reasoning models (DeepSeek-R1, etc). The answer is verifiable — is the equation right?
                Reward is automatic and binary. No human needed, but only works when correctness is objectively checkable.
              </p>
            </div>
            <div className="border-l border-white/8 pl-4">
              <div className="text-(--ollie-cyan) font-semibold text-sm mb-2">Ollie (supervised)</div>
              <p className="text-white/40 text-xs leading-relaxed">
                Labelled training data — each photo has a known identity. No reward model, no human preference ranking.
                Just: "this photo is person #4,217 out of 85,742." The loss function handles the rest.
              </p>
            </div>
          </div>
        </div>

        <Callout>
          When you give Ollie feedback, it isn&apos;t RLHF — your correction is treated as a labelled training example,
          the same as the original dataset. It goes straight into the next training run without a reward model in between.
        </Callout>
      </Section>

      {/* 03 ── CNNs */}
      <Section id="s-cnn" num="03" title="Why not just use a regular network on pixels?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          A 112×112 image has 37,632 pixels. A regular neural network would connect every pixel to every neuron in the
          first layer — millions of weights before it&apos;s even seen the whole image. That&apos;s slow, uses huge amounts of memory,
          and doesn&apos;t generalise — if you move an eye one pixel to the right, the network sees a completely different image.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          A <strong className="text-white/85">Convolutional Neural Network (CNN)</strong> solves this with a different idea:
          instead of looking at all pixels at once, slide a small window (usually 3×3 pixels) across the image and look for
          a specific pattern inside that window. The same window — the same weights — gets applied at every position.
          This is called a <strong className="text-white/85">kernel</strong> or filter.
        </p>

        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {[
            { label: "Kernel detects edges", desc: "A 3×3 kernel tuned to detect vertical edges fires strongly wherever brightness changes left-to-right. One kernel, used at every position." },
            { label: "Stack kernels for shapes", desc: "Layer 2 applies kernels to Layer 1's output. Now it detects curves and corners made of edges. Each layer sees larger, more complex patterns." },
            { label: "Deep layers see faces", desc: "By layer 10+, individual kernels respond to eyes, noses, hairlines. The network has built a hierarchy of features, from pixels up to faces." },
          ].map(({ label, desc }) => (
            <div key={label} className="p-5 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="text-white font-bold text-sm mb-2">{label}</div>
              <div className="text-white/40 text-sm leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>

        <p className="text-white/60 text-base leading-relaxed mb-6">
          Between groups of conv layers, a <strong className="text-white/85">stride-2 convolution</strong> halves the spatial
          resolution — 112→56→28→14→7 pixels — while doubling the number of feature channels.
          You trade spatial detail for richer feature representation. By the final stage, you have a 7×7 grid of
          512 feature maps: each cell knows what&apos;s in that small patch of the face.
        </p>
        <Callout>
          CNNs are why modern face recognition works. The same approach powers image search, medical scans,
          self-driving car vision, and satellite imagery analysis — anything where spatial patterns matter.
        </Callout>
      </Section>
    </div>
  )
}

function ArchitectureScroll2() {
  return (
    <div>
      {/* 04 ── Ollie's actual network */}
      <Section id="s-backbone" num="04" title="Ollie's exact network: SphereFaceNet">
        <div className="flex flex-col lg:flex-row gap-12 items-start">
          <div className="flex-1">
            <p className="text-white/60 text-base leading-relaxed mb-6">
              Ollie uses a network called <strong className="text-white/85">SphereFaceNet</strong> (sphere20) —
              a 20-layer CNN purpose-built for face recognition. It was designed to balance accuracy and speed:
              deep enough to learn fine-grained facial detail, fast enough to run on a consumer GPU.
            </p>
            <p className="text-white/60 text-base leading-relaxed mb-6">
              It has four stages, each starting with a stride-2 conv that halves the image size,
              followed by residual blocks. The block counts are <Pill>[1, 2, 4, 1]</Pill> —
              more blocks in the middle where the most complex feature combinations happen.
            </p>
            <p className="text-white/60 text-base leading-relaxed mb-6">
              After the four stages, the 7×7×512 feature map is flattened into 25,088 numbers.
              A single fully-connected layer compresses this to <strong className="text-white/85">512 numbers</strong> —
              the face fingerprint. Then <Pill>BatchNorm</Pill> stabilises the values,
              and <Pill>L2-norm</Pill> scales the vector onto a unit sphere.
              Every face is now a point on the same sphere, comparable by distance.
            </p>

            <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-4 font-mono text-xs mb-6 space-y-2">
              <div className="text-white/25 tracking-widest mb-2">ARCHITECTURE SUMMARY</div>
              <div className="flex gap-3"><span className="text-(--ollie-cyan) w-16">layer1</span><span className="text-white/50">3 → 64ch, stride-2, 1 block   (112→56)</span></div>
              <div className="flex gap-3"><span className="text-(--ollie-cyan) w-16">layer2</span><span className="text-white/50">64 → 128ch, stride-2, 2 blocks  (56→28)</span></div>
              <div className="flex gap-3"><span className="text-(--ollie-cyan) w-16">layer3</span><span className="text-white/50">128 → 256ch, stride-2, 4 blocks  (28→14)</span></div>
              <div className="flex gap-3"><span className="text-(--ollie-cyan) w-16">layer4</span><span className="text-white/50">256 → 512ch, stride-2, 1 block   (14→7)</span></div>
              <div className="flex gap-3"><span className="text-(--ollie-cyan) w-16">fc</span><span className="text-white/50">25,088 → 512 (flatten + linear)</span></div>
              <div className="flex gap-3"><span className="text-(--ollie-cyan) w-16">output</span><span className="text-white/50">BN(512) → L2-norm → unit sphere</span></div>
            </div>

            <p className="text-white/60 text-base leading-relaxed mb-6">
              Instead of ReLU (which kills negative signals entirely), sphere20 uses
              <strong className="text-white/85"> PReLU</strong> — a version where the slope for negative values is learned
              rather than fixed at zero. This preserves more information through deep layers.
            </p>
            <p className="text-white/60 text-base leading-relaxed">
              Training uses <strong className="text-white/85">CosFace loss</strong> — the network is trained to classify
              which of 85,742 identities each photo belongs to, with a cosine margin that forces it to be confidently
              right, not just barely right. This trains much faster and more accurately than trying to learn from pairs.
            </p>
          </div>
          <div className="shrink-0 w-full lg:w-[300px]">
            <EmbeddingSpaceViz />
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-10">
          {[
            { label: "Parameters",       value: "24.5M",  sub: "trainable weights" },
            { label: "Training dataset", value: "MS1MV2", sub: "85k identities" },
            { label: "Training photos",  value: "5.8M",   sub: "pre-aligned 112×112" },
            { label: "Fingerprint size", value: "512-d",  sub: "per face" },
          ].map((s) => (
            <div key={s.label} className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 text-center">
              <div className="text-(--ollie-cyan) font-black text-2xl mb-1">{s.value}</div>
              <div className="text-white/70 text-xs font-semibold">{s.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{s.sub}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* 05 ── What it physically measures */}
      <Section id="s-features" num="05" title="What does it physically measure on your face?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Alongside the 512-d neural embedding, Ollie extracts 32 direct geometric measurements.
          These are the same things a forensic artist or biometric researcher would measure — ratios and distances
          between specific facial landmarks, not abstract neural activations.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          On GPU, this uses <strong className="text-white/85">InsightFace</strong> — Meta&apos;s open-source face analysis library,
          which runs a separate detection model to find your face and place 68 landmark points on it.
          On CPU (when no GPU is available), it falls back to
          <strong className="text-white/85"> MediaPipe</strong> (Google&apos;s face mesh model), which places 478 points.
          More points = more precise measurements.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          All measurements are ratios — distance divided by face width, for example — so they&apos;re independent
          of how close you were to the camera or how large your photo is.
        </p>

        <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden mb-6">
          <div className="px-5 py-3 border-b border-white/5 flex gap-4 text-xs text-white/25 font-mono">
            <span className="w-7">#</span>
            <span className="w-36">name</span>
            <span>what it means in plain English</span>
          </div>
          <div className="px-5 pb-2">
            <FeatureRow index="0"     name="eye_dist"     plain="Distance between your eyes, relative to face width" />
            <FeatureRow index="1–2"   name="ear_L / ear_R" plain="Eye aspect ratio — how open each eye is" />
            <FeatureRow index="3–8"   name="iris_*"        plain="Iris colour and texture: hue, saturation, lightness, variance per eye" />
            <FeatureRow index="9"     name="face_ratio"    plain="Face height ÷ face width — oval vs round" />
            <FeatureRow index="10"    name="jaw_w"         plain="Jaw width relative to face width" />
            <FeatureRow index="12–13" name="nose_w / nose_h" plain="Width and height of nose relative to face" />
            <FeatureRow index="14–16" name="lip_*"         plain="Mouth width, lip height, and openness" />
            <FeatureRow index="17–19" name="hair_H/S/V"    plain="Hair colour — hue, saturation, brightness (HSV)" />
            <FeatureRow index="20–22" name="skin_L/a/b"    plain="Skin tone in LAB colour space (perceptually uniform)" />
            <FeatureRow index="23–27" name="*_ratio"       plain="Proportions between forehead, eyes, nose, lips, and chin" />
            <FeatureRow index="28"    name="age_norm"      plain="Estimated age scaled 0–1" />
            <FeatureRow index="29"    name="gender_score"  plain="Predicted gender from facial structure" />
          </div>
        </div>

        <p className="text-white/60 text-base leading-relaxed">
          These 32 values are used for <strong className="text-white/85">re-ranking</strong>.
          After FAISS returns the 200 nearest neural embeddings, anything with a very different age,
          gender, or skin tone gets pushed down the list. The neural network finds candidates;
          the geometric check filters out obvious errors.
        </p>
      </Section>

      {/* 06 ── Library stack */}
      <Section id="s-stack" num="06" title="Every library Ollie uses and why">
        <p className="text-white/60 text-base leading-relaxed mb-8">
          From reading a JPEG off disk to returning a similarity percentage — here&apos;s every tool in the chain,
          in the order it&apos;s actually used.
        </p>

        <div className="space-y-3">
          {[
            {
              lib: "PIL / Pillow",
              pkg: "pip install pillow",
              role: "Image loading",
              desc: "Opens JPEG, PNG, or WEBP files and converts them to RGB pixel arrays. The first thing that runs on your uploaded photo. Also generates black placeholder images for corrupt files.",
            },
            {
              lib: "torchvision.transforms",
              pkg: "torchvision",
              role: "Preprocessing",
              desc: "Resizes to 112×112, randomly flips horizontally during training (the only augmentation used — MS1MV2 images are pre-aligned), converts to a float tensor, and normalises pixels from [0,255] to [-1,1].",
            },
            {
              lib: "InsightFace",
              pkg: "pip install insightface",
              role: "GPU face detection",
              desc: "Detects and crops the face region, places 68 landmark points on it (eyes, nose, mouth, jaw), and extracts a pre-aligned 112×112 crop. Runs on GPU when available.",
            },
            {
              lib: "MediaPipe",
              pkg: "pip install mediapipe",
              role: "CPU face detection fallback",
              desc: "Google's lightweight face mesh model. Places 478 landmark points. Used automatically when no GPU is available. Slightly slower but runs on any machine.",
            },
            {
              lib: "PyTorch (torch)",
              pkg: "pip install torch",
              role: "Neural network core",
              desc: "Provides tensors (GPU-accelerated arrays), automatic differentiation for backpropagation, nn.Module (the base class for all layers), and the training loop primitives.",
            },
            {
              lib: "torch.amp",
              pkg: "built into PyTorch",
              role: "Automatic Mixed Precision",
              desc: "Runs convolutions and matrix multiplications in FP16 (half precision), which uses the GPU's Tensor Cores and doubles throughput. Loss computation stays in FP32 to prevent underflow. GradScaler prevents small gradients from vanishing.",
            },
            {
              lib: "torch.optim.SGD",
              pkg: "built into PyTorch",
              role: "Optimiser",
              desc: "Stochastic Gradient Descent with momentum 0.9 and weight decay 5×10⁻⁴. Momentum accumulates gradient history to push through flat regions. Weight decay penalises large weights to prevent overfitting.",
            },
            {
              lib: "torch.optim.MultiStepLR",
              pkg: "built into PyTorch",
              role: "Learning rate schedule",
              desc: "Drops the learning rate by ×0.1 at epochs 10, 20, and 25. This is the standard schedule for face recognition: learn fast early, make small corrections at the end.",
            },
            {
              lib: "NumPy",
              pkg: "pip install numpy",
              role: "Array operations + evaluation",
              desc: "LFW 10-fold cross-validation runs entirely in NumPy — threshold search, accuracy calculation, fold splitting. Also used for handling FAISS search results.",
            },
            {
              lib: "FAISS",
              pkg: "pip install faiss-gpu",
              role: "Vector similarity search",
              desc: "Meta's library for searching billions of vectors fast. Stores all celebrity face embeddings as a flat index. When you upload a photo, your 512-d vector is compared against every stored vector in ~10ms.",
            },
            {
              lib: "KaggleHub",
              pkg: "pip install kagglehub",
              role: "Dataset download",
              desc: "Downloads MS1MV2 (15.5GB) and LFW (200MB) directly from Kaggle into a local cache. Handles authentication and resumable downloads.",
            },
          ].map(({ lib, pkg, role, desc }) => (
            <div key={lib} className="flex gap-4 p-5 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="shrink-0 w-36">
                <div className="text-white font-semibold text-sm">{lib}</div>
                <div className="text-white/25 text-xs font-mono mt-0.5">{pkg}</div>
                <div className="text-(--ollie-cyan) text-xs mt-1">{role}</div>
              </div>
              <div className="text-white/40 text-sm leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* 07 ── Full pipeline */}
      <Section id="s-pipeline" num="07" title="From photo to result: the full pipeline">
        <p className="text-white/60 text-base leading-relaxed mb-8">
          Every time you upload a photo, this exact sequence runs:
        </p>

        <div className="flex flex-col gap-3 mb-8">
          {[
            {
              step: "1",
              label: "Face detection",
              detail: "InsightFace (GPU) or MediaPipe (CPU) locates your face in the image and crops it to a 112×112 aligned patch.",
              tech: "insightface / mediapipe",
            },
            {
              step: "2",
              label: "Geometric feature extraction",
              detail: "32 ratios and measurements are computed from 68–478 landmark points: eye distance, jaw width, nose proportions, iris colour, skin tone, estimated age.",
              tech: "numpy + custom geometry",
            },
            {
              step: "3",
              label: "Neural embedding",
              detail: "The 112×112 crop is normalised to [-1,1], run through SphereFaceNet's 20 layers, flattened, FC-projected to 512 dimensions, BN-normalised, and L2-projected onto the unit sphere.",
              tech: "torch + SphereFaceNet",
            },
            {
              step: "4",
              label: "FAISS nearest-neighbour search",
              detail: "Your 512-d vector is compared against every celebrity embedding in the FAISS index. The 200 closest are returned in ~10ms.",
              tech: "faiss",
            },
            {
              step: "5",
              label: "Geometric re-ranking",
              detail: "The 200 candidates are re-scored by combining neural distance with geometric feature distance. Large discrepancies in age, gender, or skin tone reduce the score.",
              tech: "numpy weighted scoring",
            },
            {
              step: "6",
              label: "Top 5 returned",
              detail: "The 5 best-scoring matches are shown with a similarity percentage. 95%+ means the embeddings are very close on the unit sphere; 60% means they share broad features but differ in detail.",
              tech: "gradio frontend",
            },
          ].map(({ step, label, detail, tech }) => (
            <div key={step} className="flex gap-4 p-5 rounded-2xl bg-white/[0.03] border border-white/8">
              <div className="w-8 h-8 rounded-full border border-white/15 flex items-center justify-center text-white/40 font-mono text-xs shrink-0 mt-0.5">
                {step}
              </div>
              <div className="flex-1">
                <div className="text-white font-semibold text-sm mb-1">{label}</div>
                <div className="text-white/40 text-sm leading-relaxed mb-2">{detail}</div>
                <div className="text-(--ollie-cyan) text-xs font-mono opacity-60">{tech}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* 08 ── Feedback */}
      <Section id="s-feedback" num="08" title="How does Ollie get better after it's deployed?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          The model isn&apos;t frozen after training. When you mark a match as wrong, that correction is stored as a
          labelled training pair. The next time the model is retrained, it sees your feedback alongside the original
          85k-identity dataset — and adjusts itself so it won&apos;t make the same mistake on faces like yours.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <div className="text-white font-bold text-sm mb-3">What happens when you submit feedback</div>
            <ol className="text-white/40 text-sm leading-relaxed list-decimal list-inside space-y-2">
              <li>Your face + the wrong celebrity are saved as a pair</li>
              <li>Labelled as &quot;these are different people&quot;</li>
              <li>Added to the training data for the next run</li>
              <li>CosFace loss pushes those embeddings apart</li>
              <li>Every future user benefits from the correction</li>
            </ol>
          </div>
          <div className="p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <div className="text-white font-bold text-sm mb-3">Why this isn&apos;t RLHF</div>
            <p className="text-white/40 text-sm leading-relaxed">
              RLHF requires a separate reward model, preference rankings, and PPO training — a pipeline used by
              large language models. Ollie&apos;s feedback is simpler and more direct: your correction is just a new
              training example with a hard label. No reward model, no subjective scoring.
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

      {/* 09 ── Training numbers */}
      <Section id="s-training" num="09" title="How long did it take to train?">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Training a neural network is not like installing software. It&apos;s a process that runs for days,
          showing the network millions of examples and adjusting billions of weights, one batch at a time.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          Ollie trains on <strong className="text-white/85">MS1MV2</strong> — a dataset of 5.8 million face photos
          across 85,742 identities, all pre-aligned to 112×112. Each epoch sees every single photo once,
          in a different random order. The <Pill>CosFace</Pill> loss treats each photo as a classification problem:
          which of 85,742 people is this? Getting the right answer — confidently — is what teaches the network
          to make good embeddings.
        </p>
        <p className="text-white/60 text-base leading-relaxed mb-6">
          With AMP (Automatic Mixed Precision) on a GeForce RTX 4060 Ti, a full 35-epoch run takes roughly
          1.2 days. On a GTX 1650 (no Tensor Cores, 4GB VRAM) it would take approximately 10–12 days.
          After training, the model is evaluated on <strong className="text-white/85">LFW</strong> —
          6,000 face pairs split into 10 folds, with the threshold tuned on 9 folds and tested on the remaining 1.
          The reference benchmark for sphere20 on MS1MV2 is <strong className="text-white/85">~99% LFW accuracy</strong>.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Target LFW accuracy", value: "~99%",   sub: "sphere20 + MS1MV2 reference" },
            { label: "Training identities",  value: "85,742", sub: "MS1MV2 dataset" },
            { label: "Training photos",      value: "5.8M",   sub: "per epoch, reshuffled" },
            { label: "Fingerprint size",     value: "512-d",  sub: "unit-sphere embedding" },
          ].map((s) => (
            <div key={s.label} className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 text-center">
              <div className="text-(--ollie-cyan) font-black text-2xl mb-1">{s.value}</div>
              <div className="text-white/70 text-xs font-semibold">{s.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{s.sub}</div>
            </div>
          ))}
        </div>

        <p className="text-white/50 text-sm leading-relaxed">
          The optimiser is <Pill>SGD</Pill> with momentum 0.9 and weight decay 5×10⁻⁴,
          with a <Pill>MultiStepLR</Pill> schedule that drops by ×0.1 at epochs 10, 20, and 25.
          Batch size is 512. Four DataLoader workers pre-fetch the next batch while the GPU is busy with the current one.
          AMP runs convolutions in FP16 (88 TFLOPS on 4060 Ti vs 22 TFLOPS FP32) — roughly a 3× throughput gain.
        </p>
      </Section>

      {/* 10 ── Blog */}
      <Section id="s-learnmore" num="10" title="Go deeper in the blog">
        <p className="text-white/60 text-base leading-relaxed mb-6">
          These articles expand on each concept above, with more detail on the algorithms and research behind Ollie.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            {
              slug: "siamese-neural-networks-explained",
              title: "Siamese Networks Explained",
              desc: "Why Ollie uses two identical networks in parallel, and how shared weights enable similarity learning.",
            },
            {
              slug: "contrastive-loss-explained",
              title: "Contrastive Loss Explained",
              desc: "The loss function that teaches the network what facial similarity means at a mathematical level.",
            },
            {
              slug: "vggface2-explained",
              title: "VGGFace2 Explained",
              desc: "The training dataset behind Ollie, with 3.3 million images across 9,131 identities.",
            },
            {
              slug: "resnet-face-recognition",
              title: "ResNet for Face Recognition",
              desc: "Why residual connections are so effective for deep face recognition networks.",
            },
            {
              slug: "arcface-explained",
              title: "ArcFace Explained",
              desc: "The margin-based loss function that achieves state-of-the-art face recognition accuracy.",
            },
            {
              slug: "transfer-learning-explained",
              title: "Transfer Learning Explained",
              desc: "How pre-training on millions of faces gives Ollie a head start on new tasks.",
            },
          ].map(({ slug, title, desc }) => (
            <Link
              key={slug}
              href={`/blog/${slug}`}
              className="block p-5 rounded-2xl bg-white/[0.03] border border-white/8 hover:border-white/20 hover:bg-white/[0.04] transition-all group"
            >
              <h3 className="text-sm font-bold text-white mb-1.5 group-hover:text-sky-400 transition-colors leading-snug">
                {title}
              </h3>
              <p className="text-white/40 text-xs leading-relaxed">{desc}</p>
            </Link>
          ))}
        </div>
      </Section>
    </div>
  )
}

// ── page ──────────────────────────────────────────────────────────────────────

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
          From biological neurons to 512-dimensional face fingerprints.
        </motion.p>
      </section>

      {/* First three sections: neurons, learning, CNNs */}
      <ArchitectureScroll />

      {/* 3D interactive network — placed here after CNNs are explained */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="px-6 pb-4 max-w-6xl mx-auto border-t border-white/5 pt-16"
      >
        <div className="mb-6">
          <span className="text-(--ollie-cyan) font-mono text-xs tracking-widest font-bold block mb-1">INTERACTIVE</span>
          <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">
            Explore the network live
          </h2>
          <p className="text-white/40 text-sm mt-3">
            Hover or click any node to see what each layer detects. Each column is one layer of SphereFaceNet.
          </p>
        </div>
        <NeuralDeepViz />
      </motion.section>

      {/* Remaining sections: architecture, libraries, pipeline, feedback, stats, blog */}
      <ArchitectureScroll2 />

      <Footer />
    </main>
  )
}
