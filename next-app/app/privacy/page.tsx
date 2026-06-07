import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"

export const metadata = {
  title: "Privacy Policy",
  description: "Ollie privacy policy. Learn how we handle your data and images.",
  alternates: { canonical: "/privacy" },
}

export default function PrivacyPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <div className="max-w-3xl mx-auto px-6 pt-32 pb-20">
        <h1 className="text-4xl font-black text-white mb-2 tracking-tight">Privacy Policy</h1>
        <p className="text-white/30 text-sm mb-12">Last updated: June 2026</p>

        <div className="prose prose-invert max-w-none space-y-10 text-white/60 leading-relaxed">

          <section>
            <h2 className="text-xl font-bold text-white mb-3">1. Overview</h2>
            <p>Ollie (&quot;we&quot;, &quot;our&quot;, &quot;us&quot;) is an AI-powered celebrity face matching tool. We are committed to protecting your privacy. This policy explains what data we collect, how we use it, and your rights.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">2. Images You Upload</h2>
            <p>Photos you upload to Ollie are processed entirely for the purpose of generating a facial embedding (a mathematical representation of your face). <strong className="text-white/80">We do not store, save, or retain your photos</strong> after processing. Images are held in memory only for the duration of the matching request and discarded immediately after.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">3. Data We Collect</h2>
            <ul className="list-disc list-inside space-y-2">
              <li><strong className="text-white/80">Feedback data:</strong> If you voluntarily submit match feedback (correct/incorrect), we store only the pair result — not your photo.</li>
              <li><strong className="text-white/80">Usage data:</strong> Standard server logs (IP address, browser type, pages visited) may be collected for security and performance monitoring.</li>
              <li><strong className="text-white/80">Cookies:</strong> We use minimal session cookies necessary for the service to function. We do not use advertising or tracking cookies.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">4. How We Use Your Data</h2>
            <p>We use collected data solely to:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>Provide and improve the face matching service</li>
              <li>Retrain the AI model using anonymised feedback pairs</li>
              <li>Monitor for abuse and security threats</li>
            </ul>
            <p className="mt-3">We do not sell, rent, or share your data with third parties for marketing purposes.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">5. Children&apos;s Privacy</h2>
            <p>Ollie is not intended for use by children under the age of 13. We do not knowingly collect personal data from children under 13. If you believe a child has submitted data, contact us and we will delete it promptly.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">6. Your Rights</h2>
            <p>Depending on your location, you may have the right to:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>Access personal data we hold about you</li>
              <li>Request deletion of your data</li>
              <li>Object to processing</li>
              <li>Data portability</li>
            </ul>
            <p className="mt-3">To exercise any of these rights, contact us at the address below.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">7. Security</h2>
            <p>We implement reasonable technical and organisational measures to protect data. However, no internet transmission is 100% secure, and we cannot guarantee absolute security.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">8. Changes to This Policy</h2>
            <p>We may update this policy periodically. Changes will be posted on this page with an updated date. Continued use of Ollie after changes constitutes acceptance.</p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-3">9. Contact</h2>
            <p>For privacy-related questions or requests, contact us at <a href="mailto:rubixion76@gmail.com" className="text-sky-400 hover:underline">rubixion76@gmail.com</a>.</p>
          </section>

        </div>
      </div>
      <Footer />
    </main>
  )
}
