'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// SVG Icons matching Webotixs brand & platform designs
function ChatIcon() {
  return (
    <svg
      width="28"
      height="28"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.3"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22z" />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  )
}

function WhatsAppIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.04 2c-5.46 0-9.91 4.45-9.91 9.91 0 1.75.46 3.45 1.32 4.95L2.05 22l5.25-1.38c1.45.79 3.08 1.21 4.74 1.21 5.46 0 9.91-4.45 9.91-9.91 0-2.65-1.03-5.14-2.9-7.01A9.816 9.816 0 0 0 12.04 2zm.01 1.67c2.2 0 4.26.86 5.82 2.42a8.225 8.225 0 0 1 2.41 5.83c0 4.54-3.7 8.24-8.24 8.24-1.48 0-2.93-.4-4.2-1.15l-.3-.18-3.12.82.83-3.04-.2-.31a8.196 8.196 0 0 1-1.26-4.38c0-4.54 3.7-8.24 8.24-8.24zm4.52 11.66c-.25-.13-1.47-.72-1.7-.81-.23-.08-.39-.13-.56.13-.17.25-.64.81-.79.97-.14.17-.29.19-.54.06-.25-.13-1.06-.39-2.02-1.25-.75-.67-1.26-1.5-1.41-1.75-.15-.25-.02-.39.11-.51.11-.11.25-.29.37-.44.13-.14.17-.25.25-.41.08-.17.04-.31-.02-.44-.06-.13-.56-1.34-.76-1.84-.2-.49-.4-.42-.56-.43h-.48c-.17 0-.44.06-.67.31-.23.25-.87.85-.87 2.07 0 1.22.89 2.4 1.01 2.57.13.17 1.75 2.67 4.24 3.75.59.26 1.05.41 1.41.53.59.19 1.13.16 1.56.1.48-.07 1.47-.6 1.68-1.18.21-.58.21-1.07.15-1.18-.06-.12-.22-.19-.47-.32z" />
    </svg>
  )
}

function GoogleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 15.96 0 12.48 0 5.8 0 0 5.4 0 12s5.8 12 12.48 12c3.6 0 6.32-1.187 8.44-3.413 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z" />
    </svg>
  )
}

function TrustpilotIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
    </svg>
  )
}

function ClutchIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
    </svg>
  )
}

function InstagramIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.1"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="20" height="20" x="2" y="2" rx="5" ry="5" />
      <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z" />
      <line x1="17.5" x2="17.51" y1="6.5" y2="6.5" />
    </svg>
  )
}

function FacebookIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
    </svg>
  )
}

function YouTubeIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
    </svg>
  )
}

function ArrowUpIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  )
}

export default function FixedFloatingWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setIsOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleScrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
    setIsOpen(false)
  }

  // Items stacked from top to bottom (WhatsApp closest to trigger button)
  const items = [
    {
      id: 'top',
      labelNode: <span>Back to top</span>,
      ariaLabel: 'Back to top',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-gray-900 shadow-md',
      icon: <ArrowUpIcon />,
      onClick: handleScrollToTop,
    },
    {
      id: 'youtube',
      labelNode: <span>YouTube</span>,
      ariaLabel: 'YouTube',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-red-600 shadow-md',
      icon: <YouTubeIcon />,
      href: 'https://www.youtube.com/@WebotixsLTD',
    },
    {
      id: 'facebook',
      labelNode: <span>Facebook</span>,
      ariaLabel: 'Facebook',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-[#1877F2] shadow-md',
      icon: <FacebookIcon />,
      href: 'https://www.facebook.com/profile.php?id=61585230680319',
    },
    {
      id: 'instagram',
      labelNode: <span>Instagram</span>,
      ariaLabel: 'Instagram',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-[#E4405F] shadow-md',
      icon: <InstagramIcon />,
      href: 'https://www.instagram.com/webotixs?igsh=dzRhNjV6cGptNXZp',
    },
    {
      id: 'clutch',
      labelNode: (
        <span className="flex items-center gap-1.5">
          <span>Clutch Reviews</span>
          <span className="bg-red-50 text-red-600 font-bold text-xs px-2 py-0.5 rounded-full border border-red-200">
            4.9★
          </span>
        </span>
      ),
      ariaLabel: 'Clutch Reviews 4.9 Stars',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-red-500 text-white shadow-md',
      icon: <ClutchIcon />,
      href: 'https://clutch.co/profile/webotixs',
    },
    {
      id: 'trustpilot',
      labelNode: (
        <span className="flex items-center gap-1.5">
          <span>Trustpilot</span>
          <span className="bg-emerald-50 text-emerald-600 font-bold text-xs px-2 py-0.5 rounded-full border border-emerald-200">
            5.0★
          </span>
        </span>
      ),
      ariaLabel: 'Trustpilot Verified Reviews 5.0 Stars',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-emerald-500 text-white shadow-md',
      icon: <TrustpilotIcon />,
      href: 'https://www.trustpilot.com/review/webotixs.com',
    },
    {
      id: 'google',
      labelNode: (
        <span className="flex items-center gap-1.5">
          <span>Google Review</span>
          <span className="bg-blue-50 text-blue-600 font-bold text-xs px-2 py-0.5 rounded-full border border-blue-200">
            5.0★
          </span>
        </span>
      ),
      ariaLabel: 'Google Business Reviews 5.0 Stars',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-blue-500 text-white shadow-md',
      icon: <GoogleIcon />,
      href: 'https://g.page/r/Cbbz0-7qGKZVEBM/review',
    },
    {
      id: 'whatsapp',
      labelNode: (
        <span className="flex items-center gap-1.5 font-bold text-emerald-700">
          <span>Need help?</span>
          <span className="bg-emerald-100/90 text-emerald-800 text-xs px-2 py-0.5 rounded-full border border-emerald-300">
            WhatsApp
          </span>
        </span>
      ),
      ariaLabel: 'Need help? Chat on WhatsApp',
      pillClass: 'bg-white text-gray-900 shadow-md',
      circleClass: 'bg-[#25D366] text-white shadow-md',
      icon: <WhatsAppIcon />,
      href: 'https://wa.me/12089055973',
    },
  ]

  return (
    <aside
      ref={containerRef}
      className="fixed bottom-5 right-4 sm:bottom-7 sm:right-7 z-50 flex flex-col items-end pointer-events-auto"
      aria-label="Quick Connect & Reviews Menu"
    >
      {/* Expanded Menu Stack */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex flex-col items-end gap-2.5 mb-3 max-h-[calc(100vh-100px)] overflow-y-auto pr-1 no-scrollbar"
          >
            {items.map((item, index) => {
              const delay = (items.length - 1 - index) * 0.03
              const content = (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 12, scale: 0.88 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.88 }}
                  transition={{ duration: 0.2, delay }}
                  className="flex items-center justify-end gap-2.5 group cursor-pointer"
                >
                  {/* Pill label on the left */}
                  <span
                    className={`px-3.5 py-1.5 sm:px-4 sm:py-2 rounded-full text-xs sm:text-sm leading-none whitespace-nowrap transition-transform duration-200 group-hover:scale-105 select-none ${item.pillClass}`}
                  >
                    {item.labelNode}
                  </span>

                  {/* Circular icon button on the right */}
                  <div
                    className={`w-11 h-11 sm:w-12 sm:h-12 rounded-full flex items-center justify-center transition-all duration-200 group-hover:scale-110 active:scale-95 ${item.circleClass}`}
                  >
                    {item.icon}
                  </div>
                </motion.div>
              )

              if (item.href) {
                return (
                  <a
                    key={item.id}
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={item.ariaLabel}
                    className="focus:outline-none"
                    onClick={() => setIsOpen(false)}
                  >
                    {content}
                  </a>
                )
              }

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={item.onClick}
                  aria-label={item.ariaLabel}
                  className="focus:outline-none"
                >
                  {content}
                </button>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Trigger Button: Webotixs Signature Blue-to-Cyan Gradient (NO YELLOW) */}
      <motion.button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close menu' : 'Open quick connect and reviews menu'}
        aria-expanded={isOpen}
        whileTap={{ scale: 0.92 }}
        className={`w-14 h-14 rounded-full flex items-center justify-center shadow-xl transition-all duration-300 focus:outline-none cursor-pointer border ${
          isOpen
            ? 'bg-[#0D1224] text-cyan-400 border-cyan-500/50 hover:bg-[#151D35] hover:border-cyan-400'
            : 'bg-gradient-to-r from-[#2563EB] to-[#06B6D4] text-white border-white/25 shadow-[0_0_25px_rgba(37,99,235,0.55)] hover:shadow-[0_0_35px_rgba(6,182,212,0.7)] hover:scale-105'
        }`}
      >
        <AnimatePresence mode="wait" initial={false}>
          {isOpen ? (
            <motion.div
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <CloseIcon />
            </motion.div>
          ) : (
            <motion.div
              key="chat"
              initial={{ rotate: 90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: -90, opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <ChatIcon />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>
    </aside>
  )
}
