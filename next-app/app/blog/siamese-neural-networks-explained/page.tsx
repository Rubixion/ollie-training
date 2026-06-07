import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export const metadata = {
  title: "Siamese Neural Networks Explained Simply",
  description: "Ollie uses a Siamese network - the same architecture used in fraud detection and medical imaging. Here is what makes it special for comparing two faces.",
  alternates: { canonical: "/blog/siamese-neural-networks-explained" },
}

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Siamese Neural Networks Explained Simply",
  "description": "Ollie uses a Siamese network - the same architecture used in fraud detection and medical imaging. Here is what makes it special for comparing two faces.",
  "datePublished": "2026-04-01",
  "dateModified": "2026-06-07",
  "author": { "@type": "Organization", "name": "Ollie" },
  "publisher": { "@type": "Organization", "name": "Ollie", "url": "https://ollieai.app" },
  "mainEntityOfPage": { "@type": "WebPage", "@id": "https://ollieai.app/blog/siamese-neural-networks-explained" },
}

export default function SiameseNetworksPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <article className="max-w-2xl mx-auto px-6 pt-32 pb-20">
        <Link
          href="/blog"
          className="inline-flex items-center gap-2 text-white/30 text-sm hover:text-white/60 transition-colors mb-10"
        >
          <ArrowLeft size={14} /> Blog
        </Link>

        <div className="mb-10">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-[10px] font-bold tracking-widest uppercase text-sky-400/70">Deep Dive</span>
            <span className="text-white/20 text-xs">April 2026</span>
            <span className="text-white/20 text-xs">5 min read</span>
          </div>
          <h1 className="text-3xl md:text-4xl font-black text-white tracking-tight leading-tight mb-4">
            Siamese Neural Networks Explained Simply
          </h1>
          <p className="text-white/50 text-base leading-relaxed">
            Most image classifiers answer "what is this?" Ollie answers "are these two faces the same person?"
            That requires a completely different architecture - and a clever one.
          </p>
        </div>

        <div className="prose prose-invert prose-sm max-w-none space-y-8 text-white/70 leading-relaxed">

          <section>
            <h2 className="text-xl font-bold text-white mb-3">The problem with standard classifiers</h2>
            <p>
              A standard image classifier - like the kind used to identify cats versus dogs - works by
              training on millions of labelled examples and learning to map an image to a category.
              This works well when you have a fixed, known set of categories.
            </p>
            <p className="mt-3">
              But face matching is different. You cannot train a classifier on every possible human face
              because new people appear every day. The set of "categories" is open-ended and constantly growing.
              You need an approach that generalises to faces the network has never seen before.
            </p>
            <p className="mt-3">
              This is exactly the problem Siamese networks were designed to solve.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">What is a Siamese network?</h2>
            <p>
              A Siamese network is two identical neural networks that share the same weights - meaning every
              parameter update to one automatically applies to the other. They process two inputs simultaneously
              and produce two outputs (called embeddings). The final step is comparing those two outputs.
            </p>
            <p className="mt-3">
              The name comes from Siamese twins - two distinct things joined together. In this case, two
              identical networks sharing the same internal structure.
            </p>
            <p className="mt-3">
              For Ollie, the two inputs are two face photos. Each network produces a 256-number vector (the
              facial fingerprint). The distance between these two vectors is how the model decides whether
              the faces belong to the same person or different people.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">How training works</h2>
            <p>
              Training a Siamese network requires pairs of images, not individual labelled images. Specifically:
            </p>
            <ul className="list-disc pl-5 mt-3 space-y-2">
              <li><strong className="text-white">Positive pairs</strong> - two photos of the same person. The network should learn to produce similar vectors for these.</li>
              <li><strong className="text-white">Negative pairs</strong> - two photos of different people. The network should produce different vectors for these.</li>
            </ul>
            <p className="mt-3">
              The loss function (called contrastive loss) penalises the network when it produces similar vectors
              for a negative pair, or different vectors for a positive pair. Over millions of training examples,
              the network learns what features matter for distinguishing faces.
            </p>
            <p className="mt-3">
              Ollie was trained on roughly 3.3 million face pairs from the VGGFace2 dataset - one of the largest
              publicly available face recognition datasets, containing over 9,000 identities.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Why shared weights matter</h2>
            <p>
              The key insight of the architecture is that both networks must be identical. If they had different
              weights, one network might learn to extract completely different features than the other. Comparing
              the outputs would be meaningless - like comparing a height measurement to a weight measurement.
            </p>
            <p className="mt-3">
              Shared weights guarantee that both networks are applying the same transformation to both inputs.
              The embedding space is consistent, so distances in that space mean something real.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Where else Siamese networks are used</h2>
            <p>
              The architecture is well-suited to any problem where you need to compare two things rather
              than classify one thing. Common applications include:
            </p>
            <ul className="list-disc pl-5 mt-3 space-y-2">
              <li><strong className="text-white">Fraud detection</strong> - comparing a new transaction signature to known fraudulent patterns</li>
              <li><strong className="text-white">Medical imaging</strong> - detecting whether a follow-up scan shows progression of a condition compared to a baseline</li>
              <li><strong className="text-white">Document verification</strong> - comparing a submitted signature to a reference signature</li>
              <li><strong className="text-white">Duplicate detection</strong> - finding near-identical images or articles across large datasets</li>
              <li><strong className="text-white">One-shot learning</strong> - recognising a new category from a single example, used in robotics and assistive technology</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">The threshold problem</h2>
            <p>
              One non-obvious challenge with Siamese networks is setting the decision threshold - the distance
              at which you say "these are the same person" versus "these are different people".
            </p>
            <p className="mt-3">
              Set it too tight and the model rejects genuine matches (same person, different photo). Set it
              too loose and it incorrectly matches different people who share similar features.
            </p>
            <p className="mt-3">
              Ollie calibrates this threshold automatically during training by testing against a held-out set
              of known pairs. The current threshold is tuned to minimise false matches - so when you do get
              a high-confidence result, it is reliable.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Why build it from scratch?</h2>
            <p>
              Off-the-shelf face recognition APIs exist and work well. Building a custom Siamese network
              from scratch is significantly more work. So why do it?
            </p>
            <p className="mt-3">
              The main reason is control. A custom model means full control over what the network is optimised
              for, what data it trains on, how it handles celebrity-specific features, and how it improves from
              user feedback. It also means the model can be fine-tuned specifically for the celebrity matching
              task rather than general identity verification.
            </p>
            <p className="mt-3">
              The secondary reason is transparency. When something goes wrong, you can look inside and understand
              why - something that is impossible with a black-box API.
            </p>
          </section>

          <div className="pt-8 border-t border-white/8">
            <p className="text-white/30 text-sm mb-4">See the AI in action</p>
            <a
              href="/match"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-sky-500/20 border border-sky-500/30 text-sky-300 text-sm font-semibold hover:bg-sky-500/30 transition-colors"
            >
              Find your celebrity match
            </a>
          </div>
        </div>
      </article>
      <Footer />
    </main>
  )
}
