import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { Mail, MessageSquare, Code2 } from "lucide-react"

export const metadata = {
  title: "Contact",
  description: "Get in touch with the Ollie team. Questions, feedback, partnerships, or press enquiries.",
  alternates: { canonical: "/contact" },
}

export default function ContactPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <div className="max-w-2xl mx-auto px-6 pt-32 pb-20">
        <h1 className="text-4xl font-black text-white mb-3 tracking-tight">Contact</h1>
        <p className="text-white/40 text-base leading-relaxed mb-14">
          Have a question, found a bug, or want to collaborate? Reach out below.
        </p>

        <div className="space-y-4">
          <a
            href="mailto:rubixion76@gmail.com"
            className="flex items-center gap-5 p-6 rounded-2xl bg-white/[0.03] border border-white/8 hover:border-white/20 transition-all group"
          >
            <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center shrink-0">
              <Mail size={18} className="text-white/40 group-hover:text-white/70 transition-colors" />
            </div>
            <div>
              <p className="text-white font-semibold text-sm">Email</p>
              <p className="text-white/40 text-sm">rubixion76@gmail.com</p>
            </div>
          </a>

          <a
            href="/feedback"
            className="flex items-center gap-5 p-6 rounded-2xl bg-white/[0.03] border border-white/8 hover:border-white/20 transition-all group"
          >
            <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center shrink-0">
              <MessageSquare size={18} className="text-white/40 group-hover:text-white/70 transition-colors" />
            </div>
            <div>
              <p className="text-white font-semibold text-sm">Match Feedback</p>
              <p className="text-white/40 text-sm">Help improve the AI by rating match results</p>
            </div>
          </a>

          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-5 p-6 rounded-2xl bg-white/[0.03] border border-white/8 hover:border-white/20 transition-all group"
          >
            <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center shrink-0">
              <Code2 size={18} className="text-white/40 group-hover:text-white/70 transition-colors" />
            </div>
            <div>
              <p className="text-white font-semibold text-sm">GitHub</p>
              <p className="text-white/40 text-sm">Open source contributions and bug reports</p>
            </div>
          </a>
        </div>

        <p className="text-white/20 text-xs mt-12">
          We typically respond within 48 hours. For legal requests, please use the email above and include &quot;Legal&quot; in the subject line.
        </p>
      </div>
      <Footer />
    </main>
  )
}
