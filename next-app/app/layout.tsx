import type { Metadata } from "next"
import { Outfit, JetBrains_Mono } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { AuthProvider } from "@/components/auth-provider"
import { AuthModal } from "@/components/auth-modal"
import { cn } from "@/lib/utils"

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["300", "400", "500", "600", "700", "800", "900"],
})

const fontMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600"],
})

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://ollieai.app"

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: "Ollie - Find Your Celebrity Lookalike",
    template: "%s | Ollie",
  },
  description: "Upload your photo and discover which celebrity you most resemble. Free AI-powered celebrity face matching using deep learning and facial recognition technology.",
  keywords: ["celebrity lookalike", "celebrity face match", "AI face recognition", "who do I look like", "celebrity twin finder", "facial recognition AI", "celebrity doppelganger"],
  openGraph: {
    type: "website",
    siteName: "Ollie",
    title: "Ollie - Find Your Celebrity Lookalike",
    description: "Upload your photo and discover which celebrity you most resemble. Free AI-powered celebrity face matching.",
    url: siteUrl,
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Ollie - Find Your Celebrity Lookalike",
    description: "Upload your photo and discover which celebrity you most resemble. Free AI-powered celebrity face matching.",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-image-preview": "large" },
  },
  alternates: { canonical: siteUrl },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn("dark antialiased", fontMono.variable, outfit.variable)}
    >
      <body suppressHydrationWarning>
        <ThemeProvider defaultTheme="dark" enableSystem={false}>
          <AuthProvider>
            {children}
            <AuthModal />
          </AuthProvider>
        </ThemeProvider>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              "name": "Ollie",
              "url": siteUrl,
              "description": "Ollie builds AI-powered facial recognition tools, starting with a celebrity face matching app backed by a custom neural network.",
              "contactPoint": { "@type": "ContactPoint", "contactType": "customer support", "url": `${siteUrl}/contact` },
              "knowsAbout": ["Facial recognition", "Deep learning", "Celebrity face matching", "Neural networks"],
            }),
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebApplication",
              "name": "Ollie",
              "url": siteUrl,
              "description": "AI-powered celebrity face matching. Upload your photo and discover which celebrity you most resemble.",
              "applicationCategory": "EntertainmentApplication",
              "operatingSystem": "Web",
              "browserRequirements": "Requires JavaScript",
              "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
              "creator": { "@type": "Organization", "name": "Ollie", "url": siteUrl },
              "featureList": ["Celebrity face matching", "AI facial recognition", "Siamese neural network", "Real-time results"],
            }),
          }}
        />
      </body>
    </html>
  )
}
