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

// â”€â”€ Code block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="font-mono text-sm bg-transparent/60 border border-white/8 rounded-xl p-6 text-slate-200 overflow-x-auto leading-relaxed whitespace-pre-wrap">
      {children}
    </pre>
  )
}

// â”€â”€ Content card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function ContentCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-8">
      {children}
    </div>
  )
}

// â”€â”€ Feature row (for the 32-dim table) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function FeatureRow({ index, name, description }: { index: string; name: string; description: string }) {
  return (
    <div className="flex gap-4 py-2 border-b border-white/5 last:border-0 font-mono text-xs">
      <span className="text-(--ollie-cyan) w-8 shrink-0">[{index}]</span>
      <span className="text-white/80 w-40 shrink-0">{name}</span>
      <span className="text-white/40">{description}</span>
    </div>
  )
}

// â”€â”€ Section header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function SectionHeader({ num, title }: { num: string; title: string }) {
  return (
    <div className="mb-10">
      <span className="text-(--ollie-cyan) font-mono text-xs tracking-widest font-bold">{num}</span>
      <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight mt-1">{title}</h2>
    </div>
  )
}

// â”€â”€ Architecture scroll sections â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function ArchitectureScroll() {
  return (
    <div className="border-t border-white/5 mt-8">

      {/* Section 01 â€” Residual CNN Backbone */}
      <motion.div
        id="s-backbone"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto scroll-mt-20"
      >
        <SectionHeader num="01" title="Residual CNN Backbone" />
        <div className="flex flex-col lg:flex-row gap-12 items-start">
          {/* Left: Face feature visualization */}
          <div className="shrink-0 flex flex-col items-center gap-4">
            <div className="rounded-2xl overflow-hidden border border-white/10 bg-transparent/40 p-2">
              <FaceFeatureViz size={300} />
            </div>
            <p className="text-white/30 text-xs text-center max-w-[280px]">
              32 geometric features extracted from facial landmarks
            </p>
          </div>
          {/* Right: Description */}
          <div className="flex-1">
            <ContentCard>
              <p className="text-white/60 text-sm leading-relaxed mb-5">
                The backbone is a custom 8-block ResNet trained from scratch on face pairs â€” not pretrained on ImageNet.
                This matters because ImageNet-biased features don&apos;t generalize to face identity.
              </p>
              <CodeBlock>{`Architecture: 8 ResBlocks, 3Ã—3 convolutions
Channels: 3 â†’ 64 â†’ 128 â†’ 256 â†’ 512
Stride-2 downsampling: 96Ã—96 â†’ 48Ã—48 â†’ 24Ã—24 â†’ 12Ã—12 â†’ 6Ã—6
Global average pooling: 6Ã—6Ã—512 â†’ 512-d vector
Residual connections: prevent vanishing gradients via skip paths
BatchNorm after each conv: normalizes activations
ReLU activations: non-linearity without saturation
Total backbone params: ~2.1M learnable weights`}</CodeBlock>
            </ContentCard>
          </div>
        </div>
      </motion.div>

      {/* Section 02 â€” Siamese Architecture */}
      <motion.div
        id="s-siamese"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto border-t border-white/5 scroll-mt-20"
      >
        <SectionHeader num="02" title="Siamese Twin Networks" />
        <ContentCard>
          <p className="text-white/60 text-sm leading-relaxed mb-5">
            The same backbone runs twice â€” once for each face â€” but with <strong className="text-white/90">one shared set of weights</strong>.
            This forces the network to learn a consistent transformation: the same identity maps to the same region
            of embedding space regardless of which &quot;twin&quot; processes it.
          </p>
          <CodeBlock>{`Two identical networks, ONE set of weights.
f(xâ‚) and f(xâ‚‚) use the same W, b at every layer.

Why? Metric learning: d(f(xâ‚), f(xâ‚‚)) must be small
for same-person pairs, large for different-person pairs.

Weight sharing enforces: "same transformation applied
to both faces" â€” the distance is meaningful.

Embedding: L2-normalized 256-d vector.
cos(f(xâ‚), f(xâ‚‚)) = f(xâ‚)Â·f(xâ‚‚) / (â€–f(xâ‚)â€–Â·â€–f(xâ‚‚)â€–)`}</CodeBlock>
        </ContentCard>
      </motion.div>

      {/* Section 03 â€” Contrastive Loss */}
      <motion.div
        id="s-contrastive"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto border-t border-white/5 scroll-mt-20"
      >
        <SectionHeader num="03" title="Contrastive Loss Function" />
        <ContentCard>
          <p className="text-white/60 text-sm leading-relaxed mb-5">
            The loss has two terms: one that pulls same-person pairs to distance â‰ˆ 0, and one that
            pushes different-person pairs to distance â‰¥ 2.0. The margin of 2.0 gives the network a
            large enough separation target to be discriminative.
          </p>
          <CodeBlock>{`L = (1-y)Â·Â½Â·dÂ² + yÂ·Â½Â·max(0, margin-d)Â²

where:
  y = 1 if same person, 0 if different
  d = â€–f(xâ‚) - f(xâ‚‚)â€–â‚‚  (L2 distance in embedding space)
  margin = 2.0  (min separation for negative pairs)

Loss pulls same-person pairs toward d â‰ˆ 0
Loss pushes different-person pairs toward d â‰¥ 2.0

BCE head: additional sigmoid(|eâ‚-eâ‚‚|Â·W+b) â†’ similarity score
Training: Adam, lr=1e-4, weight_decay=1e-4
Data: LFW + VGGFace2 (3.3M images, 9,131 identities)`}</CodeBlock>
        </ContentCard>
      </motion.div>

      {/* Section 04 â€” Geometric Features */}
      <motion.div
        id="s-features"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto border-t border-white/5 scroll-mt-20"
      >
        <SectionHeader num="04" title="32-Dimensional Geometric Feature Vector" />
        <ContentCard>
          <p className="text-white/60 text-sm leading-relaxed mb-5">
            Beyond the CNN embedding, a handcrafted 32-dim vector captures interpretable face geometry
            using InsightFace GPU detection and MediaPipe&apos;s 478-point face mesh. These features re-rank
            the top FAISS candidates with physically meaningful penalties.
          </p>
          <div className="flex flex-col">
            <FeatureRow index="0" name="eye_dist" description="inter-ocular distance / face_w" />
            <FeatureRow index="1" name="ear_left" description="eye aspect ratio (left eye openness)" />
            <FeatureRow index="2" name="ear_right" description="eye aspect ratio (right eye)" />
            <FeatureRow index="3" name="iris_hue_L" description="iris HSV hue (left)" />
            <FeatureRow index="4" name="iris_hue_R" description="iris HSV hue (right)" />
            <FeatureRow index="5" name="iris_sat_L" description="iris saturation" />
            <FeatureRow index="6" name="iris_sat_R" description="iris saturation" />
            <FeatureRow index="7" name="iris_var_L" description="iris texture variance" />
            <FeatureRow index="8" name="iris_var_R" description="iris texture variance" />
            <FeatureRow index="9" name="face_ratio" description="face_h / face_w" />
            <FeatureRow index="10" name="jaw_w" description="jaw width / face_w" />
            <FeatureRow index="11" name="face_shape" description="face_w / face_h (reciprocal)" />
            <FeatureRow index="12" name="nose_w" description="nose width / face_w" />
            <FeatureRow index="13" name="nose_h" description="nose height / face_h" />
            <FeatureRow index="14" name="lip_ratio" description="lip width / face_w" />
            <FeatureRow index="15" name="lip_h_ratio" description="lip height / face_h" />
            <FeatureRow index="16" name="lip_open" description="mouth openness" />
            <FeatureRow index="17" name="hair_H" description="hair region HSV hue" />
            <FeatureRow index="18" name="hair_S" description="hair saturation" />
            <FeatureRow index="19" name="hair_V" description="hair value (brightness)" />
            <FeatureRow index="20" name="skin_L" description="skin LAB lightness" />
            <FeatureRow index="21" name="skin_a" description="skin LAB green-red axis" />
            <FeatureRow index="22" name="skin_b" description="skin LAB blue-yellow axis" />
            <FeatureRow index="23" name="forehead_ratio" description="forehead height / face_h" />
            <FeatureRow index="24" name="eye_nose_ratio" description="(nose_y - eye_y) / face_h" />
            <FeatureRow index="25" name="nose_mouth_ratio" description="(mouth_y - nose_y) / face_h" />
            <FeatureRow index="26" name="mouth_chin_ratio" description="(chin_y - mouth_y) / face_h" />
            <FeatureRow index="27" name="hair_length" description="hair length score (0-1)" />
            <FeatureRow index="28" name="age_norm" description="predicted age / 100 (InsightFace)" />
            <FeatureRow index="29" name="gender_score" description="1.0=male, 0.0=female (InsightFace)" />
            <FeatureRow index="30" name="face_area_ratio" description="log1p(face_area / img_area)" />
            <FeatureRow index="31" name="forehead_L" description="forehead LAB lightness" />
            <div className="mt-6 pt-6 border-t border-white/10">
              <CodeBlock>{`Re-ranking weights: ageÃ—3.0, genderÃ—4.0, skin_LÃ—5.0,
                    hair_HÃ—2.0, face_shapeÃ—1.5
Extraction: InsightFace GPU (det_10g + genderage ONNX)
            + MediaPipe FaceMesh (478 landmarks, CPU fallback)`}</CodeBlock>
            </div>
          </div>
        </ContentCard>
      </motion.div>

      {/* Section 05 â€” FAISS */}
      <motion.div
        id="s-faiss"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto border-t border-white/5 scroll-mt-20"
      >
        <SectionHeader num="05" title="Vector Search with FAISS" />
        <ContentCard>
          <p className="text-white/60 text-sm leading-relaxed mb-5">
            FAISS (Facebook AI Similarity Search) handles embedding lookups at scale.
            An <code className="text-white/80 bg-white/5 px-1 rounded">IndexFlatL2</code> does exact brute-force L2 search
            on all 17K+ indexed embeddings in milliseconds â€” returning 200 candidates for re-ranking.
          </p>
          <CodeBlock>{`Index type: IndexFlatL2 (exact L2 search)
Complexity: O(nÂ·d) per query, d=256 dimensions
Speed: ~100Ã— faster than numpy dot product

Search pipeline:
1. Extract query embedding: f(x_query) âˆˆ â„Â²âµâ¶
2. FAISS search: top-200 candidates by L2 distance
3. Re-rank by geometric features + penalties
4. Return top-5 with similarity scores

Re-ranking formula:
  penalty = 5.0Â·Î”skin + 2.0Â·Î”hair + 1.5Â·Î”shape
          + 3.0Â·Î”age + 4.0Â·Î”gender
  final_score = embed_dist + penalty

Scale: handles 10M+ embeddings on a single machine
Current dataset: ~17K images indexed at startup`}</CodeBlock>
        </ContentCard>
      </motion.div>

      {/* Section 06 â€” Feedback */}
      <motion.div
        id="s-feedback"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto border-t border-white/5 scroll-mt-20"
      >
        <SectionHeader num="06" title="Human Feedback Loop" />
        <ContentCard>
          <p className="text-white/60 text-sm leading-relaxed mb-5">
            Every match you mark correct or wrong gets stored as a labeled pair and injected directly
            into the next training run â€” treated identically to official LFW labels. No reward model, no PPO â€”
            just direct supervised signal from real users.
          </p>
          <CodeBlock>{`Feedback mechanism: user marks match as correct/wrong
Storage: feedback_pairs.csv (path1, path2, label)

Training integration:
  train_pairs = official_lfw_csv
             + generated_filesystem (max 5000)
             + VGGFace2 (max 50000)
             + celeb_pairs
             + human_feedback  â† yours

Effect: feedback pairs treated identically to
labeled training pairs. Hard negatives from
wrong predictions improve discriminability.

This is NOT RLHF in the LLM sense â€” it is direct
supervised signal injected into contrastive training.`}</CodeBlock>
        </ContentCard>
        <div className="mt-6 flex">
          <Link
            href="/feedback"
            className="group flex items-center gap-2 px-6 py-3 rounded-full bg-(--ollie-cyan)/10 border border-(--ollie-cyan)/30 text-(--ollie-cyan) text-sm font-semibold hover:bg-(--ollie-cyan)/20 transition-all"
          >
            Submit Feedback
            <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </motion.div>

      {/* Section 07 â€” Training */}
      <motion.div
        id="s-training"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="py-24 px-6 max-w-6xl mx-auto border-t border-white/5 scroll-mt-20"
      >
        <SectionHeader num="07" title="End-to-End Training" />
        <ContentCard>
          <CodeBlock>{`Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
Scheduler: ReduceLROnPlateau (patience=15, factor=0.5)
Loss: BCE(sigmoid(|e1-e2|Â·W)) + Contrastive(margin=2.0)
Batch: 32 pairs, shuffled each epoch
Max epochs: 200

Hardware: GTX 1650 4GB VRAM (training)
          GTX 1650 (InsightFace inference, ONNX Runtime)

Backbone: Custom ResNet (not pretrained)
  â†’ Ensures embeddings are face-specific, not ImageNet-biased
  â†’ Trained from scratch on face pairs only

Best so far: 83.5% test accuracy @ epoch 90 (LFW pairs)
Current: adapting to VGGFace2 distribution (~72% LFW)
         Expected to surpass 88% at convergence

Checkpoint format: { epoch, model_state, optimizer_state,
                     scheduler_state, best_acc, threshold }`}</CodeBlock>
        </ContentCard>

        {/* Stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
          {[
            { label: "Best Accuracy", value: "83.5%", sub: "LFW test pairs" },
            { label: "Identities", value: "9,131", sub: "VGGFace2 + LFW" },
            { label: "Training Images", value: "3.3M", sub: "pair samples" },
            { label: "Embedding Dim", value: "256-d", sub: "L2 normalized" },
          ].map((stat) => (
            <div
              key={stat.label}
              className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 text-center"
            >
              <div className="text-(--ollie-cyan) font-black text-2xl mb-1">{stat.value}</div>
              <div className="text-white/70 text-xs font-semibold">{stat.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}

// â”€â”€ Page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export default function NeuralPage() {
  return (
    <main className="relative bg-transparent text-white">
      <DottedSurface />
      <Nav />

      {/* Hero: Particle text */}
      <section className="relative min-h-[40vh] flex flex-col items-center justify-center pt-24 pb-12 px-6 text-center">
        <div className="w-full max-w-5xl mx-auto mb-6">
          <ParticleTextEffect
            words={["CNN BACKBONE", "SIAMESE NET", "CONTRASTIVE LOSS", "EMBEDDING SPACE", "FACE FEATURES", "FAISS SEARCH"]}
          />
        </div>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="text-white/40 text-xs tracking-widest uppercase"
        >
          The Architecture Behind Ollie
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
