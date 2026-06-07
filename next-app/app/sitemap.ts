import { MetadataRoute } from "next"

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://ollieai.app"

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: siteUrl, lastModified: "2026-06-07", changeFrequency: "weekly", priority: 1.0 },
    { url: `${siteUrl}/match`, lastModified: "2026-06-07", changeFrequency: "monthly", priority: 0.9 },
    { url: `${siteUrl}/ai`, lastModified: "2026-06-07", changeFrequency: "monthly", priority: 0.8 },
    { url: `${siteUrl}/search`, lastModified: "2026-06-07", changeFrequency: "monthly", priority: 0.6 },
    { url: `${siteUrl}/blog`, lastModified: "2026-06-07", changeFrequency: "weekly", priority: 0.8 },
    { url: `${siteUrl}/blog/how-face-recognition-works`, lastModified: "2026-06-07", changeFrequency: "yearly", priority: 0.7 },
    { url: `${siteUrl}/blog/find-your-celebrity-lookalike`, lastModified: "2026-06-07", changeFrequency: "yearly", priority: 0.7 },
    { url: `${siteUrl}/blog/siamese-neural-networks-explained`, lastModified: "2026-06-07", changeFrequency: "yearly", priority: 0.6 },
    { url: `${siteUrl}/about`, lastModified: "2026-06-07", changeFrequency: "monthly", priority: 0.5 },
    { url: `${siteUrl}/contact`, lastModified: "2026-06-07", changeFrequency: "yearly", priority: 0.4 },
    { url: `${siteUrl}/privacy`, lastModified: "2026-06-07", changeFrequency: "yearly", priority: 0.3 },
    { url: `${siteUrl}/terms`, lastModified: "2026-06-07", changeFrequency: "yearly", priority: 0.3 },
  ]
}
