import { notFound } from "next/navigation"
import type { Metadata } from "next"
import Link from "next/link"
import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { allPosts, getPost } from "@/lib/blog-posts"
import { ArrowLeft, ArrowRight, Clock, User, Calendar, ChevronRight } from "lucide-react"

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  return allPosts.map((post) => ({ slug: post.slug }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const post = getPost(slug)
  if (!post) return {}

  const title = post.title.length <= 60 ? post.title : post.title.slice(0, 57) + "..."
  const description = post.excerpt.length <= 160 ? post.excerpt : post.excerpt.slice(0, 157) + "..."
  const canonical = `https://ollieai.app/blog/${post.slug}`
  const ogImage = `https://ollieai.app/og-blog.jpg`

  return {
    title,
    description,
    keywords: post.keywords?.join(", "),
    authors: [{ name: post.author ?? "Ollie Research Team" }],
    alternates: { canonical },
    openGraph: {
      title,
      description,
      url: canonical,
      siteName: "Ollie",
      images: [{ url: ogImage, width: 1200, height: 630, alt: title }],
      type: "article",
      publishedTime: post.isoDate,
      authors: [post.author ?? "Ollie Research Team"],
      tags: post.keywords,
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [ogImage],
      site: "@ollieai",
    },
  }
}

function TableOfContents({ headings }: { headings: string[] }) {
  if (headings.length < 2) return null
  return (
    <nav className="mb-10 p-5 rounded-xl bg-white/[0.03] border border-white/8">
      <p className="text-[10px] font-bold tracking-widest uppercase text-white/30 mb-3">In this article</p>
      <ol className="space-y-1.5">
        {headings.map((h, i) => (
          <li key={i}>
            <a
              href={`#h-${i}`}
              className="text-sm text-white/50 hover:text-sky-400 transition-colors leading-snug block"
            >
              {h}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  )
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params
  const post = getPost(slug)
  if (!post) notFound()

  const h2s = post.sections?.filter((s) => s.h2).map((s) => s.h2!) ?? []
  const relatedPosts = (post.relatedSlugs ?? [])
    .map((s) => getPost(s))
    .filter(Boolean) as typeof allPosts

  const articleSchema = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: post.title,
    description: post.excerpt,
    author: { "@type": "Organization", name: post.author ?? "Ollie Research Team" },
    publisher: {
      "@type": "Organization",
      name: "Ollie",
      url: "https://ollieai.app",
    },
    datePublished: post.isoDate,
    dateModified: post.isoDate,
    url: `https://ollieai.app/blog/${post.slug}`,
    keywords: post.keywords?.join(", "),
  }

  const faqSchema =
    post.faqs && post.faqs.length > 0
      ? {
          "@context": "https://schema.org",
          "@type": "FAQPage",
          mainEntity: post.faqs.map((faq) => ({
            "@type": "Question",
            name: faq.q,
            acceptedAnswer: { "@type": "Answer", text: faq.a },
          })),
        }
      : null

  const breadcrumbSchema = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Home", item: "https://ollieai.app" },
      { "@type": "ListItem", position: 2, name: "Blog", item: "https://ollieai.app/blog" },
      { "@type": "ListItem", position: 3, name: post.title, item: `https://ollieai.app/blog/${post.slug}` },
    ],
  }

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleSchema) }}
      />
      {faqSchema && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(faqSchema) }}
        />
      )}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbSchema) }}
      />

      <main className="relative min-h-screen bg-black">
        <Nav />

        <div className="max-w-3xl mx-auto px-6 pt-28 pb-24">
          {/* Breadcrumb */}
          <nav className="flex items-center gap-1.5 text-xs text-white/30 mb-10">
            <Link href="/" className="hover:text-white/60 transition-colors">Home</Link>
            <ChevronRight size={12} />
            <Link href="/blog" className="hover:text-white/60 transition-colors">Blog</Link>
            <ChevronRight size={12} />
            <span className="text-white/50 truncate max-w-[200px]">{post.title}</span>
          </nav>

          {/* Header */}
          <header className="mb-10">
            <div className="flex flex-wrap items-center gap-3 mb-5">
              <span className="text-[10px] font-bold tracking-widest uppercase text-sky-400/80 bg-sky-400/10 px-2.5 py-1 rounded-full">
                {post.category}
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-black text-white leading-tight mb-5 tracking-tight">
              {post.title}
            </h1>
            <p className="text-white/50 text-lg leading-relaxed mb-6">{post.excerpt}</p>
            <div className="flex flex-wrap items-center gap-4 text-xs text-white/30 border-t border-white/8 pt-5">
              <span className="flex items-center gap-1.5">
                <User size={12} />
                {post.author ?? "Ollie Research Team"}
              </span>
              <span className="flex items-center gap-1.5">
                <Calendar size={12} />
                {post.date}
              </span>
              <span className="flex items-center gap-1.5">
                <Clock size={12} />
                {post.readTime}
              </span>
            </div>
          </header>

          {/* Table of contents */}
          <TableOfContents headings={h2s} />

          {/* Article body */}
          <article className="prose prose-invert prose-sm max-w-none">
            {post.sections && post.sections.length > 0 ? (
              post.sections.map((section, i) => (
                <section key={i}>
                  {section.h2 && (
                    <h2
                      id={`h-${h2s.indexOf(section.h2)}`}
                      className="text-xl font-bold text-white mt-10 mb-4 scroll-mt-24"
                    >
                      {section.h2}
                    </h2>
                  )}
                  {section.paragraphs.map((p, j) => (
                    <p
                      key={j}
                      className="text-white/60 leading-relaxed mb-4"
                      dangerouslySetInnerHTML={{ __html: p }}
                    />
                  ))}
                </section>
              ))
            ) : (
              <p className="text-white/50 leading-relaxed">
                This article is available in full on the Ollie platform. Visit{" "}
                <Link href="/match" className="text-sky-400 hover:underline">
                  ollieai.app/match
                </Link>{" "}
                to try face matching today.
              </p>
            )}
          </article>

          {/* FAQ section */}
          {post.faqs && post.faqs.length > 0 && (
            <section className="mt-14">
              <h2 className="text-xl font-bold text-white mb-6">Frequently Asked Questions</h2>
              <div className="space-y-4">
                {post.faqs.map((faq, i) => (
                  <div key={i} className="p-5 rounded-xl bg-white/[0.03] border border-white/8">
                    <h3 className="text-sm font-semibold text-white mb-2">{faq.q}</h3>
                    <p className="text-white/50 text-sm leading-relaxed">{faq.a}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* CTA */}
          <section className="mt-14 p-7 rounded-2xl bg-sky-400/5 border border-sky-400/20">
            <p className="text-[10px] font-bold tracking-widest uppercase text-sky-400/60 mb-2">Try it yourself</p>
            <h2 className="text-xl font-bold text-white mb-2">Find your celebrity match</h2>
            <p className="text-white/50 text-sm mb-5">
              Upload a photo and see which celebrity you most resemble, powered by Ollie&apos;s AI face recognition.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/match"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-sky-400 text-black text-sm font-bold hover:bg-sky-300 transition-colors"
              >
                Find my celebrity match <ArrowRight size={14} />
              </Link>
              {(post.category === "Machine Learning" || post.category === "Deep Dive" || post.category === "Technology") && (
                <Link
                  href="/ai"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/8 border border-white/15 text-white/70 text-sm font-semibold hover:bg-white/12 hover:text-white transition-all"
                >
                  How Ollie works <ArrowRight size={14} />
                </Link>
              )}
              {(post.slug.includes("feedback") || post.slug.includes("improving") || post.slug.includes("accuracy") || post.slug.includes("bias") || post.slug.includes("confidence") || post.slug.includes("understanding")) && (
                <Link
                  href="/feedback"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/8 border border-white/15 text-white/70 text-sm font-semibold hover:bg-white/12 hover:text-white transition-all"
                >
                  Give feedback <ArrowRight size={14} />
                </Link>
              )}
            </div>
          </section>

          {/* Related posts */}
          {relatedPosts.length > 0 && (
            <section className="mt-14">
              <h2 className="text-xl font-bold text-white mb-6">Related Articles</h2>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {relatedPosts.map((related) => (
                  <Link
                    key={related.slug}
                    href={`/blog/${related.slug}`}
                    className="block p-5 rounded-xl bg-white/[0.02] border border-white/8 hover:border-white/20 hover:bg-white/[0.04] transition-all group"
                  >
                    <span className="text-[9px] font-bold tracking-widest uppercase text-sky-400/60 block mb-2">
                      {related.category}
                    </span>
                    <h3 className="text-sm font-semibold text-white/80 group-hover:text-white transition-colors leading-snug mb-1">
                      {related.title}
                    </h3>
                    <span className="text-xs text-white/25">{related.readTime}</span>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {/* Back link */}
          <div className="mt-14 pt-8 border-t border-white/8">
            <Link
              href="/blog"
              className="inline-flex items-center gap-2 text-sm text-white/40 hover:text-white/70 transition-colors"
            >
              <ArrowLeft size={14} />
              All articles
            </Link>
          </div>
        </div>

        <Footer />
      </main>
    </>
  )
}
