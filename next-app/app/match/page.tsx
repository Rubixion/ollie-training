import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { BGPattern } from "@/components/bg-pattern"
import { CelebrityFinder } from "@/components/celebrity-finder"

export const metadata = {
  title: "Celebrity Match - Ollie",
  description: "Upload your photo and discover which celebrity you resemble using Ollie's AI face matching.",
}

export default function FindPage() {
  return (
    <main className="relative min-h-screen bg-transparent">
      <BGPattern variant="grid" mask="fade-edges" fill="rgba(255,255,255,0.04)" size={32} className="fixed" />
      <Nav />
      <div className="pt-24">
        <CelebrityFinder />
      </div>

      <div className="max-w-4xl mx-auto px-6 pb-20">
        <div className="mt-2 pt-12 border-t border-white/8">
          <p className="text-[10px] font-bold tracking-widest uppercase text-white/30 mb-3">Read the guides</p>
          <h2 className="text-2xl font-black text-white mb-2">Get your best results</h2>
          <p className="text-white/40 text-sm mb-8">
            Photo quality has a big impact on accuracy. These guides show you exactly what to do.
          </p>
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              {
                slug: "best-photo-celebrity-match",
                title: "Best photos for celebrity matching",
                desc: "What makes a photo perform well, lighting, framing, and camera settings that improve accuracy.",
              },
              {
                slug: "best-lighting-for-match",
                title: "Lighting guide for face matching",
                desc: "From overcast daylight to ring lights, ranked by how much they improve your match.",
              },
              {
                slug: "improving-ollie-results",
                title: "How to improve your results",
                desc: "Step-by-step tips covering photo conditions, accessories, and what to avoid.",
              },
            ].map(({ slug, title, desc }) => (
              <Link
                key={slug}
                href={`/blog/${slug}`}
                className="block p-5 rounded-2xl bg-white/[0.02] border border-white/8 hover:border-white/20 hover:bg-white/[0.04] transition-all group"
              >
                <h3 className="text-sm font-bold text-white mb-2 group-hover:text-sky-400 transition-colors leading-snug">
                  {title}
                </h3>
                <p className="text-white/35 text-xs leading-relaxed mb-3">{desc}</p>
                <span className="inline-flex items-center gap-1 text-white/25 text-xs group-hover:text-white/50 transition-colors">
                  Read guide <ArrowRight size={11} className="group-hover:translate-x-1 transition-transform" />
                </span>
              </Link>
            ))}
          </div>
        </div>
      </div>

      <Footer />
    </main>
  )
}
