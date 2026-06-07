"use client"

import Link from "next/link"

const product = [
  { label: "Celebrity Match", href: "/match" },
  { label: "How It Works", href: "/ai" },
  { label: "Face Search", href: "/search" },
  { label: "Feedback", href: "/feedback" },
]

const company = [
  { label: "About", href: "/about" },
  { label: "Blog", href: "/blog" },
  { label: "Contact", href: "/contact" },
]

const legal = [
  { label: "Privacy Policy", href: "/privacy" },
  { label: "Terms of Service", href: "/terms" },
]

export function Footer() {
  return (
    <footer className="border-t border-white/5 pt-14 pb-8 px-6 mt-auto">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-10 mb-12">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="text-white font-black text-xl tracking-widest hover:text-white/60 transition-colors">
              OLLIE
            </Link>
            <p className="text-white/25 text-xs leading-relaxed mt-3 max-w-[200px]">
              AI-powered celebrity face matching. Upload your photo, find your twin.
            </p>
            <p className="text-white/15 text-[10px] mt-4">Not affiliated with any celebrity.</p>
          </div>

          {/* Product */}
          <div>
            <p className="text-white/40 text-[10px] font-bold tracking-widest uppercase mb-4">Product</p>
            <ul className="flex flex-col gap-3">
              {product.map((l) => (
                <li key={l.href}>
                  <Link href={l.href} className="text-white/30 hover:text-white text-sm transition-colors">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company */}
          <div>
            <p className="text-white/40 text-[10px] font-bold tracking-widest uppercase mb-4">Company</p>
            <ul className="flex flex-col gap-3">
              {company.map((l) => (
                <li key={l.href}>
                  <Link href={l.href} className="text-white/30 hover:text-white text-sm transition-colors">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal */}
          <div>
            <p className="text-white/40 text-[10px] font-bold tracking-widest uppercase mb-4">Legal</p>
            <ul className="flex flex-col gap-3">
              {legal.map((l) => (
                <li key={l.href}>
                  <Link href={l.href} className="text-white/30 hover:text-white text-sm transition-colors">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-white/5 pt-6 flex flex-col md:flex-row items-center justify-between gap-2">
          <p className="text-white/20 text-xs">
            &copy; 2026 Ollie. Built with a Siamese Neural Network.
          </p>
          <p className="text-white/10 text-[10px]">
            For entertainment purposes only. Results are not scientifically verified.
          </p>
        </div>
      </div>
    </footer>
  )
}
