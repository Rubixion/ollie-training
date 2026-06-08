"use client"

import { useState } from "react"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { allPosts, allCategories } from "@/lib/blog-posts"

export default function BlogPage() {
  const [activeCategory, setActiveCategory] = useState("All")

  const filtered =
    activeCategory === "All"
      ? allPosts
      : allPosts.filter((p) => p.category === activeCategory)

  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <div className="max-w-4xl mx-auto px-6 pt-32 pb-20">
        <div className="mb-10">
          <h1 className="text-4xl font-black text-white mb-3 tracking-tight">Blog</h1>
          <p className="text-white/40 text-base">
            AI, facial recognition, and the tech behind Ollie. {allPosts.length} articles.
          </p>
        </div>

        {/* Category filter */}
        <div className="flex flex-wrap gap-2 mb-10">
          {allCategories.map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                activeCategory === cat
                  ? "bg-sky-400 text-black border-sky-400"
                  : "text-white/40 border-white/10 hover:border-white/30 hover:text-white/70"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="space-y-4">
          {filtered.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="block p-7 rounded-2xl bg-white/[0.02] border border-white/8 hover:border-white/20 hover:bg-white/[0.04] transition-all group"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-[10px] font-bold tracking-widest uppercase text-sky-400/70">
                  {post.category}
                </span>
                <span className="text-white/15 text-xs">{post.date}</span>
                <span className="text-white/15 text-xs">{post.readTime}</span>
              </div>
              <h2 className="text-lg font-bold text-white mb-1.5 group-hover:text-white/90 transition-colors leading-snug">
                {post.title}
              </h2>
              <p className="text-white/35 text-sm leading-relaxed mb-3">{post.excerpt}</p>
              <span className="inline-flex items-center gap-1 text-white/25 text-xs group-hover:text-white/50 transition-colors">
                Read article <ArrowRight size={11} className="group-hover:translate-x-1 transition-transform" />
              </span>
            </Link>
          ))}
        </div>

        {filtered.length === 0 && (
          <p className="text-white/30 text-sm py-10 text-center">No articles in this category yet.</p>
        )}
      </div>
      <Footer />
    </main>
  )
}
