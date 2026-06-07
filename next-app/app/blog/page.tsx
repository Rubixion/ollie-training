import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import Link from "next/link"
import { ArrowRight } from "lucide-react"

export const metadata = {
  title: "Blog",
  description: "Articles about AI, facial recognition, and the technology behind Ollie.",
  alternates: { canonical: "/blog" },
}

const posts = [
  {
    slug: "how-face-recognition-works",
    title: "How Does AI Facial Recognition Actually Work?",
    excerpt: "A plain-English breakdown of how a neural network turns a photo into a unique numerical fingerprint - and how that fingerprint is used to find your celebrity match.",
    date: "June 2026",
    readTime: "6 min read",
    category: "Technology",
  },
  {
    slug: "find-your-celebrity-lookalike",
    title: "Find Your Celebrity Lookalike: A Complete Guide",
    excerpt: "Everything you need to know about getting the best match results - lighting, angles, photo quality, and why some photos work better than others.",
    date: "May 2026",
    readTime: "4 min read",
    category: "Guide",
  },
  {
    slug: "siamese-neural-networks-explained",
    title: "Siamese Neural Networks Explained Simply",
    excerpt: "Ollie uses a Siamese network - the same architecture used in fraud detection and medical imaging. Here is what makes it special for comparing two faces.",
    date: "April 2026",
    readTime: "5 min read",
    category: "Deep Dive",
  },
]

export default function BlogPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <div className="max-w-4xl mx-auto px-6 pt-32 pb-20">
        <div className="mb-14">
          <h1 className="text-4xl font-black text-white mb-3 tracking-tight">Blog</h1>
          <p className="text-white/40 text-base">AI, facial recognition, and the tech behind Ollie.</p>
        </div>

        <div className="space-y-6">
          {posts.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="block p-8 rounded-2xl bg-white/[0.02] border border-white/8 hover:border-white/20 hover:bg-white/[0.04] transition-all group"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="text-[10px] font-bold tracking-widest uppercase text-sky-400/70">{post.category}</span>
                <span className="text-white/15 text-xs">{post.date}</span>
                <span className="text-white/15 text-xs">{post.readTime}</span>
              </div>
              <h2 className="text-xl font-bold text-white mb-2 group-hover:text-white/90 transition-colors">{post.title}</h2>
              <p className="text-white/40 text-sm leading-relaxed mb-4">{post.excerpt}</p>
              <span className="inline-flex items-center gap-1 text-white/30 text-xs group-hover:text-white/60 transition-colors">
                Read article <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform" />
              </span>
            </Link>
          ))}
        </div>
      </div>
      <Footer />
    </main>
  )
}
