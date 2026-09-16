'use client'

import React, { useState, useEffect } from 'react'
import { MessageCircle, ChevronUp } from 'lucide-react'

// Custom SVGs for Clutch, Trustpilot, Google
function ClutchIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
    </svg>
  )
}

function TrustpilotIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
    </svg>
  )
}

function GoogleGIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 15.96 0 12.48 0 5.8 0 0 5.4 0 12s5.8 12 12.48 12c3.6 0 6.32-1.187 8.44-3.413 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z" />
    </svg>
  )
}

export default function FixedFloatingWidget() {
  const [showBackToTop, setShowBackToTop] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 250)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <>
      {/* DESKTOP MODE: Sleek Narrow Vertical Dock (Right-Center) */}
      <aside
        className="hidden md:flex fixed right-2.5 top-1/2 -translate-y-1/2 z-50 flex-col items-center"
        aria-label="Quick Connect Dock"
      >
        <div className="bg-[#050816]/90 border border-primary/30 backdrop-blur-xl rounded-full py-2.5 px-1.5 flex flex-col items-center gap-2.5 shadow-[0_0_20px_rgba(59,130,246,0.25)] relative group">
          {/* WhatsApp Direct */}
          <a
            href="https://wa.me/12089055973"
            target="_blank"
            rel="noopener noreferrer"
            className="group/item relative flex items-center justify-center"
            aria-label="Chat on WhatsApp"
          >
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center text-emerald-400 group-hover/item:bg-emerald-500 group-hover/item:text-white group-hover/item:scale-110 transition-all shadow-md">
              <MessageCircle size={15} />
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-emerald-400 rounded-full animate-ping" />
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-emerald-500 rounded-full border border-black" />
            </div>

            {/* Hover Tooltip */}
            <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-1.5 glass rounded-xl text-xs font-bold text-white whitespace-nowrap opacity-0 group-hover/item:opacity-100 pointer-events-none transition-all shadow-xl border border-emerald-500/30">
              WhatsApp Chat (+1 208 905-5973)
            </div>
          </a>

          <div className="w-4 h-[1px] bg-border/50" />

          {/* Clutch */}
          <a
            href="https://clutch.co/profile/webotixs"
            target="_blank"
            rel="noopener noreferrer"
            className="group/item relative flex items-center justify-center"
            aria-label="Clutch Profile"
          >
            <div className="w-8 h-8 rounded-full bg-red-500/15 border border-red-500/40 flex items-center justify-center text-red-400 group-hover/item:bg-red-600 group-hover/item:text-white group-hover/item:scale-110 transition-all shadow-md">
              <ClutchIcon size={14} />
            </div>

            {/* Hover Tooltip */}
            <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-1.5 glass rounded-xl text-xs font-bold text-white whitespace-nowrap opacity-0 group-hover/item:opacity-100 pointer-events-none transition-all shadow-xl border border-red-500/30">
              Clutch Reviews 4.9★
            </div>
          </a>

          {/* Trustpilot */}
          <a
            href="https://www.trustpilot.com/review/webotixs.com"
            target="_blank"
            rel="noopener noreferrer"
            className="group/item relative flex items-center justify-center"
            aria-label="Trustpilot Reviews"
          >
            <div className="w-8 h-8 rounded-full bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400 group-hover/item:bg-emerald-600 group-hover/item:text-white group-hover/item:scale-110 transition-all shadow-md">
              <TrustpilotIcon size={14} />
            </div>

            {/* Hover Tooltip */}
            <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-1.5 glass rounded-xl text-xs font-bold text-white whitespace-nowrap opacity-0 group-hover/item:opacity-100 pointer-events-none transition-all shadow-xl border border-emerald-500/30">
              Trustpilot 5.0★
            </div>
          </a>

          {/* Google */}
          <a
            href="https://g.page/r/Cbbz0-7qGKZVEBM/review"
            target="_blank"
            rel="noopener noreferrer"
            className="group/item relative flex items-center justify-center"
            aria-label="Google Reviews"
          >
            <div className="w-8 h-8 rounded-full bg-blue-500/15 border border-blue-500/40 flex items-center justify-center text-blue-400 group-hover/item:bg-blue-600 group-hover/item:text-white group-hover/item:scale-110 transition-all shadow-md">
              <GoogleGIcon size={14} />
            </div>

            {/* Hover Tooltip */}
            <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-1.5 glass rounded-xl text-xs font-bold text-white whitespace-nowrap opacity-0 group-hover/item:opacity-100 pointer-events-none transition-all shadow-xl border border-blue-500/30">
              Google Review 5.0★
            </div>
          </a>

          {/* Back to top */}
          {showBackToTop && (
            <>
              <div className="w-4 h-[1px] bg-border/50" />
              <button
                onClick={scrollToTop}
                className="group/item relative flex items-center justify-center"
                aria-label="Back to top"
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-primary-from to-primary-to text-white flex items-center justify-center group-hover/item:scale-110 transition-all shadow-md">
                  <ChevronUp size={16} />
                </div>
                <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-1.5 glass rounded-xl text-xs font-bold text-white whitespace-nowrap opacity-0 group-hover/item:opacity-100 pointer-events-none transition-all shadow-xl border border-primary/30">
                  Back to top
                </div>
              </button>
            </>
          )}
        </div>
      </aside>

      {/* MOBILE MODE: Compact Horizontal Bottom Dock */}
      <aside
        className="md:hidden fixed bottom-4 right-3 z-50 flex items-center gap-2"
        aria-label="Mobile Quick Dock"
      >
        <div className="bg-[#050816]/95 border border-primary/40 backdrop-blur-2xl rounded-full px-3 py-1.5 flex items-center gap-2.5 shadow-[0_0_20px_rgba(59,130,246,0.3)]">
          {/* WhatsApp Direct */}
          <a
            href="https://wa.me/12089055973"
            target="_blank"
            rel="noopener noreferrer"
            className="w-8 h-8 rounded-full bg-emerald-500 text-white flex items-center justify-center shadow-md active:scale-95 transition-transform"
            aria-label="WhatsApp"
          >
            <MessageCircle size={16} />
            <span className="sr-only">Chat on WhatsApp</span>
          </a>

          {/* Clutch */}
          <a
            href="https://clutch.co/profile/webotixs"
            target="_blank"
            rel="noopener noreferrer"
            className="w-8 h-8 rounded-full bg-red-600/90 text-white flex items-center justify-center shadow-md active:scale-95 transition-transform"
            aria-label="Clutch"
          >
            <ClutchIcon size={14} />
            <span className="sr-only">Clutch Agency Reviews</span>
          </a>

          {/* Trustpilot */}
          <a
            href="https://www.trustpilot.com/review/webotixs.com"
            target="_blank"
            rel="noopener noreferrer"
            className="w-8 h-8 rounded-full bg-emerald-600/90 text-white flex items-center justify-center shadow-md active:scale-95 transition-transform"
            aria-label="Trustpilot"
          >
            <TrustpilotIcon size={14} />
            <span className="sr-only">Trustpilot Verified Reviews</span>
          </a>

          {/* Google */}
          <a
            href="https://g.page/r/Cbbz0-7qGKZVEBM/review"
            target="_blank"
            rel="noopener noreferrer"
            className="w-8 h-8 rounded-full bg-blue-600/90 text-white flex items-center justify-center shadow-md active:scale-95 transition-transform"
            aria-label="Google"
          >
            <GoogleGIcon size={14} />
            <span className="sr-only">Google Business Reviews</span>
          </a>

          {/* Back to top on mobile */}
          {showBackToTop && (
            <button
              onClick={scrollToTop}
              className="w-8 h-8 rounded-full bg-gradient-to-r from-primary-from to-primary-to text-white flex items-center justify-center shadow-md active:scale-95 transition-transform"
              aria-label="Back to top"
            >
              <ChevronUp size={16} />
            </button>
          )}
        </div>
      </aside>
    </>
  )
}
