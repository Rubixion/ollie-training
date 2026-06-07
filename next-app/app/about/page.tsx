import AboutPageClient from "./about-client"

export const metadata = {
  title: "About",
  description: "Ollie is a free AI celebrity face matching app built from scratch with a custom-trained neural network. No third-party APIs, no stored photos.",
  alternates: { canonical: "/about" },
}

export default function AboutPage() {
  return <AboutPageClient />
}
