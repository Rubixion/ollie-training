import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export const metadata = {
  title: "Find Your Celebrity Lookalike: A Complete Guide",
  description: "Everything you need to know about getting the best celebrity match results - lighting, angles, photo quality, and why some photos work better than others.",
  alternates: { canonical: "/blog/find-your-celebrity-lookalike" },
}

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Find Your Celebrity Lookalike: A Complete Guide",
  "description": "Everything you need to know about getting the best celebrity match results - lighting, angles, photo quality, and why some photos work better than others.",
  "datePublished": "2026-05-01",
  "dateModified": "2026-06-07",
  "author": { "@type": "Organization", "name": "Ollie" },
  "publisher": { "@type": "Organization", "name": "Ollie", "url": "https://ollieai.app" },
  "mainEntityOfPage": { "@type": "WebPage", "@id": "https://ollieai.app/blog/find-your-celebrity-lookalike" },
}

export default function FindCelebrityLookalikePage() {
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
            <span className="text-[10px] font-bold tracking-widest uppercase text-sky-400/70">Guide</span>
            <span className="text-white/20 text-xs">May 2026</span>
            <span className="text-white/20 text-xs">4 min read</span>
          </div>
          <h1 className="text-3xl md:text-4xl font-black text-white tracking-tight leading-tight mb-4">
            Find Your Celebrity Lookalike: A Complete Guide
          </h1>
          <p className="text-white/50 text-base leading-relaxed">
            Getting a great celebrity match is partly down to the AI and partly down to your photo.
            Here is exactly what to do to get the most accurate result.
          </p>
        </div>

        <div className="prose prose-invert prose-sm max-w-none space-y-8 text-white/70 leading-relaxed">

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Why photo quality matters so much</h2>
            <p>
              The AI works by turning your face into a 256-number fingerprint - a compact summary of
              your bone structure, proportions, and features. If the photo is blurry, heavily filtered,
              or at an extreme angle, the fingerprint it builds is less accurate. A better photo means
              a better fingerprint, which means a more accurate match.
            </p>
            <p className="mt-3">
              Think of it like trying to identify someone through frosted glass versus a clear window.
              The AI can often work through imperfect photos, but it is working harder and making more guesses.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">The ideal photo</h2>
            <p>The single best type of photo for face matching is a front-facing portrait with:</p>
            <ul className="list-disc pl-5 mt-3 space-y-2">
              <li>Your face taking up at least 40% of the frame</li>
              <li>Even, natural lighting - no harsh shadows across one side</li>
              <li>A neutral or slight smile expression (not a wide grin that changes your face shape)</li>
              <li>Eyes open and visible</li>
              <li>No sunglasses, heavy makeup, or face-obscuring accessories</li>
              <li>A reasonably plain background - busy patterns can confuse the face detector</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Lighting: the biggest factor</h2>
            <p>
              Lighting affects face matching more than anything else. The AI was trained on celebrity press
              photos, which are almost always shot with professional, even lighting. When your photo has
              strong directional light - a window to one side, sunlight from above, a lamp behind you -
              it casts shadows that change how your face looks geometrically.
            </p>
            <p className="mt-3">
              The best light is diffuse and even. Overcast daylight outdoors, or indoors facing a large window,
              works extremely well. Avoid ring lights directly in front (they flatten your face) and candles
              or dim restaurant lighting (too little data for the AI to work with).
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Angle: front-facing wins</h2>
            <p>
              The AI handles angles, but front-facing photos consistently produce the best results. When you
              turn your head, the visible proportions of your face change - your nose appears wider or narrower,
              your jaw looks different, and one eye is partially hidden.
            </p>
            <p className="mt-3">
              A slight 3/4 angle (about 15-20 degrees off centre) is fine and is actually how most celebrity
              headshots are taken. Avoid profiles (90 degrees) and extreme low or high angles. If you want
              to test a specific look, upload a front-facing photo first to get a baseline, then try the
              other photo.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">What about filters and edits?</h2>
            <p>
              Heavy Instagram-style filters, skin smoothing, and beauty modes change your face in ways the AI
              cannot fully reverse. Extreme skin smoothing erases the texture details the AI uses to differentiate
              between similar faces. Heavy saturation or colour grading changes how the model reads skin tone
              and shadow depth.
            </p>
            <p className="mt-3">
              Light edits - a slight brightness boost, minor cropping - are fine. The general rule is: if you
              can still see the natural texture of your skin and the natural shadows on your face, the photo
              will work well.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">Why do I get different results on different photos?</h2>
            <p>
              This is normal and actually tells you something useful. The celebrities who appear across multiple
              different photos of you are probably your genuine closest matches. The ones who only appear in one
              specific photo are likely responding to that photo&apos;s specific lighting or expression rather than
              your underlying face structure.
            </p>
            <p className="mt-3">
              Try 2-3 different photos taken in different conditions and look for the names that repeat.
              Those are your real matches.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">The similarity percentage explained</h2>
            <p>
              The percentage shown next to each match is how similar your facial fingerprint is to the
              celebrity&apos;s fingerprint. 100% would mean identical - which only happens if you upload
              the exact same photo.
            </p>
            <p className="mt-3">
              In practice, anything above 75% is a strong match. 60-75% is a reasonable resemblance that
              most people would agree with. Below 60% the AI is reaching - these matches are more about
              specific shared features than overall resemblance.
            </p>
            <p className="mt-3">
              The threshold is calibrated so that roughly the top 1-2% of comparisons register as a match,
              so even a 70% result means you genuinely resemble that person more than most people do.
            </p>
          </section>

          <div className="pt-8 border-t border-white/8">
            <p className="text-white/30 text-sm mb-4">Try it yourself</p>
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
