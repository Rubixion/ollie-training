import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Blog — AI, Facial Recognition & Celebrity Matching",
  description:
    "Articles about AI facial recognition, celebrity matching, neural networks, and the science behind Ollie. 80+ in-depth guides.",
  alternates: { canonical: "https://ollieai.app/blog" },
  openGraph: {
    title: "Blog — Ollie",
    description: "Deep dives into AI facial recognition, neural networks, and celebrity face matching.",
    url: "https://ollieai.app/blog",
    siteName: "Ollie",
    type: "website",
  },
}

export default function BlogLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}
