import { Nav } from "@/components/nav"
import { Footer } from "@/components/footer"
import { Mail, MessageSquare } from "lucide-react"

export const metadata = {
  title: "Contact",
  description: "Get in touch with the Ollie team. Questions, bug reports, feedback, or anything else.",
  alternates: { canonical: "/contact" },
}

export default function ContactPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <Nav />
      <div className="max-w-xl mx-auto px-6 pt-32 pb-20">
        <div className="mb-14">
          <h1 className="text-4xl font-black text-white mb-3 tracking-tight">Get in touch</h1>
          <p className="text-white/40 text-base leading-relaxed">
            Got a question, found a bug, or want to share something? We read everything.
          </p>
        </div>

        <div className="space-y-3">
          <a
            href="mailto:rubixion76@gmail.com"
            className="flex items-start gap-5 p-6 rounded-2xl bg-white/[0.03] border border-white/8 hover:border-white/20 hover:bg-white/[0.05] transition-all group"
          >
            <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center shrink-0 mt-0.5">
              <Mail size={18} className="text-white/40 group-hover:text-white/70 transition-colors" />
            </div>
            <div>
              <p className="text-white font-semibold text-sm mb-1">Email</p>
              <p className="text-white/50 text-sm mb-1">rubixion76@gmail.com</p>
              <p className="text-white/25 text-xs">Typically replied within 48 hours.</p>
            </div>
          </a>

          <a
            href="/feedback"
            className="flex items-start gap-5 p-6 rounded-2xl bg-white/[0.03] border border-white/8 hover:border-white/20 hover:bg-white/[0.05] transition-all group"
          >
            <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center shrink-0 mt-0.5">
              <MessageSquare size={18} className="text-white/40 group-hover:text-white/70 transition-colors" />
            </div>
            <div>
              <p className="text-white font-semibold text-sm mb-1">Report a wrong match</p>
              <p className="text-white/50 text-sm mb-1">Got a result that looks off? Tell us.</p>
              <p className="text-white/25 text-xs">Every report goes straight into the next training run.</p>
            </div>
          </a>
        </div>

        <p className="text-white/20 text-xs mt-10 leading-relaxed">
          For legal or press enquiries, email us with the subject line &quot;Legal&quot; or &quot;Press&quot;.
        </p>
      </div>
      <Footer />
    </main>
  )
}
