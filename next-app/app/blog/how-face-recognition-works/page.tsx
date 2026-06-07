import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export const metadata = {
  title: "How Does AI Facial Recognition Actually Work?",
  description: "A plain-English breakdown of how a neural network turns a photo into a unique numerical fingerprint and how that fingerprint finds your celebrity match.",
  alternates: { canonical: "/blog/how-face-recognition-works" },
  openGraph: {
    title: "How Does AI Facial Recognition Actually Work?",
    description: "A plain-English breakdown of how a neural network turns a photo into a unique numerical fingerprint.",
    type: "article",
    publishedTime: "2026-06-01",
  },
}

export default function ArticlePage() {
  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "How Does AI Facial Recognition Actually Work?",
            "description": "A plain-English breakdown of how a neural network turns a photo into a unique numerical fingerprint.",
            "datePublished": "2026-06-01",
            "author": { "@type": "Organization", "name": "Ollie" },
            "publisher": { "@type": "Organization", "name": "Ollie" },
          }),
        }}
      />
      <div className="max-w-2xl mx-auto px-6 pt-32 pb-20">
        <Link href="/blog" className="inline-flex items-center gap-2 text-white/30 hover:text-white text-sm mb-10 transition-colors">
          <ArrowLeft size={14} /> Back to Blog
        </Link>

        <div className="mb-4">
          <span className="text-[10px] font-bold tracking-widest uppercase text-sky-400/70">Technology</span>
        </div>
        <h1 className="text-4xl font-black text-white mb-4 tracking-tight leading-tight">
          How Does AI Facial Recognition Actually Work?
        </h1>
        <p className="text-white/30 text-sm mb-12">June 2026 &middot; 6 min read</p>

        <div className="prose prose-invert max-w-none space-y-6 text-white/60 leading-relaxed text-base">

          <p>
            When you upload a photo to Ollie, something remarkable happens in under a second. The app does not compare pixels directly. It does not measure the distance between your eyes and call it a day. Instead, it converts your entire face into a list of 256 numbers - a <strong className="text-white/80">facial fingerprint</strong> - and then finds the celebrities whose fingerprints are closest to yours.
          </p>
          <p>
            Here is exactly how that works.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">Step 1: Your face becomes a grid of numbers</h2>
          <p>
            Every digital photo is already just numbers. A 96x96 pixel image is a grid of 9,216 pixels. Each pixel has three values (red, green, blue) ranging from 0 to 255. So your photo arrives at the AI as a block of 27,648 numbers.
          </p>
          <p>
            That raw grid is useless for comparison. Two photos of the same person taken from slightly different angles will have completely different pixel values. The AI needs to extract what is <em>structurally</em> the same across photos of the same person.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">Step 2: The convolutional layers find patterns</h2>
          <p>
            The neural network starts with several <strong className="text-white/80">convolutional layers</strong>. Think of each layer as a filter that slides across the image, looking for a specific pattern. Early layers detect low-level patterns: edges, colour gradients, corners. Later layers combine these into higher-level structures: eye sockets, nose bridges, jawlines.
          </p>
          <p>
            By the fourth convolutional block, the network is no longer seeing pixels. It is seeing <em>identity-relevant geometry</em> - the shapes and proportions that make your face yours.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">Step 3: 512 features collapse to 256 numbers</h2>
          <p>
            After the convolutional layers, a technique called <strong className="text-white/80">global average pooling</strong> compresses the spatial information into a flat vector of 512 values. A final fully connected layer then compresses this to 256 values and normalises the result so that all values together have a length of exactly 1.
          </p>
          <p>
            These 256 numbers are your facial fingerprint. They encode everything the network has learned is meaningful about face identity - in a format that is easy to compare.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">Step 4: Distance equals difference</h2>
          <p>
            Because fingerprints are normalised to unit length, comparing two faces is simply measuring the straight-line distance between two points in 256-dimensional space. Two photos of the same person will have fingerprints that are close together. Two different people will be further apart.
          </p>
          <p>
            Ollie converts this distance into a similarity percentage using a formula calibrated so that identical faces score 100% and completely unrelated faces score near 0%.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">Step 5: The Siamese structure</h2>
          <p>
            Ollie uses a <strong className="text-white/80">Siamese network</strong> - two identical copies of the same network that process your photo and a celebrity photo simultaneously. Both copies share the exact same weights, which means both fingerprints live in the same mathematical space and can be directly compared.
          </p>
          <p>
            The network was trained by showing it millions of face pairs - same person and different people - and learning to pull same-person pairs closer together while pushing different-person pairs further apart. That is contrastive learning.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">Plus geometric measurements</h2>
          <p>
            On top of the neural network fingerprint, Ollie also extracts 32 geometric measurements from your face: eye spacing, nose width, jaw shape, lip proportions, forehead height, iris colour, hair colour, and skin tone. These are fused with the neural network output before searching.
          </p>
          <p>
            This hybrid approach catches cases where two faces have similar geometry but different neural fingerprints, and vice versa.
          </p>

          <h2 className="text-2xl font-bold text-white mt-10">The search itself is instant</h2>
          <p>
            Ollie pre-computes fingerprints for every celebrity in its database and stores them in a <strong className="text-white/80">FAISS index</strong> - a library designed for high-speed nearest-neighbour search in high-dimensional space. When you upload a photo, your fingerprint is computed once and then searched against thousands of pre-computed celebrity fingerprints in about 10 milliseconds.
          </p>

          <div className="mt-12 p-6 rounded-2xl bg-white/[0.03] border border-white/8">
            <p className="text-white/60 text-sm">Want to see it in action? <Link href="/match" className="text-sky-400 hover:underline">Try Ollie now</Link> or <Link href="/ai" className="text-sky-400 hover:underline">explore the interactive architecture diagram</Link>.</p>
          </div>

        </div>
      </div>
      <Footer />
    </main>
  )
}
