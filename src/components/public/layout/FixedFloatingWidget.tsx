'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// SVG Icons matching brand designs
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
      strokeWidth="2.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  )
}

function WhatsAppIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.04 2c-5.46 0-9.91 4.45-9.91 9.91 0 1.75.46 3.45 1.32 4.95L2.05 22l5.25-1.38c1.45.79 3.08 1.21 4.74 1.21 5.46 0 9.91-4.45 9.91-9.91 0-2.65-1.03-5.14-2.9-7.01A9.816 9.816 0 0 0 12.04 2zm.01 1.67c2.2 0 4.26.86 5.82 2.42a8.225 8.225 0 0 1 2.41 5.83c0 4.54-3.7 8.24-8.24 8.24-1.48 0-2.93-.4-4.2-1.15l-.3-.18-3.12.82.83-3.04-.2-.31a8.196 8.196 0 0 1-1.26-4.38c0-4.54 3.7-8.24 8.24-8.24zm4.52 11.66c-.25-.13-1.47-.72-1.7-.81-.23-.08-.39-.13-.56.13-.17.25-.64.81-.79.97-.14.17-.29.19-.54.06-.25-.13-1.06-.39-2.02-1.25-.75-.67-1.26-1.5-1.41-1.75-.15-.25-.02-.39.11-.51.11-.11.25-.29.37-.44.13-.14.17-.25.25-.41.08-.17.04-.31-.02-.44-.06-.13-.56-1.34-.76-1.84-.2-.49-.4-.42-.56-.43h-.48c-.17 0-.44.06-.67.31-.23.25-.87.85-.87 2.07 0 1.22.89 2.4 1.01 2.57.13.17 1.75 2.67 4.24 3.75.59.26 1.05.41 1.41.53.59.19 1.13.16 1.56.1.48-.07 1.47-.6 1.68-1.18.21-.58.21-1.07.15-1.18-.06-.12-.22-.19-.47-.32z" />
    </svg>
  )
}

function InstagramIcon() {
  return (
    <svg
      width="22"
      height="22"
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
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
    </svg>
  )
}

function TikTokIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64c.298-.002.595.042.88.13V9.4a6.33 6.33 0 0 0-1-.08A6.34 6.34 0 0 0 3 15.66a6.34 6.34 0 0 0 10.82 4.49 6.27 6.27 0 0 0 1.95-4.48V8.66a8.28 8.28 0 0 0 4.82 1.48v-3.45z" />
    </svg>
  )
}

function YouTubeIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
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

  const items = [
    {
      id: 'top',
      label: 'Back to top',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-gray-900 shadow-md',
      icon: <ArrowUpIcon />,
      onClick: handleScrollToTop,
    },
    {
      id: 'youtube',
      label: 'YouTube',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-gray-900 shadow-md',
      icon: <YouTubeIcon />,
      href: 'https://youtube.com/@webotixs',
    },
    {
      id: 'tiktok',
      label: 'TikTok',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-gray-900 shadow-md',
      icon: <TikTokIcon />,
      href: 'https://www.tiktok.com/@webotixs',
    },
    {
      id: 'facebook',
      label: 'Facebook',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-gray-900 shadow-md',
      icon: <FacebookIcon />,
      href: 'https://www.facebook.com/profile.php?id=61585230680319',
    },
    {
      id: 'instagram',
      label: 'Instagram',
      pillClass: 'bg-white text-gray-900 font-semibold shadow-md',
      circleClass: 'bg-white text-gray-900 shadow-md',
      icon: <InstagramIcon />,
      href: 'https://www.instagram.com/webotixs?igsh=dzRhNjV6cGptNXZp',
    },
    {
      id: 'whatsapp',
      label: 'Need help?',
      pillClass: 'bg-[#FFBE0B] text-gray-950 font-bold shadow-md',
      circleClass: 'bg-[#25D366] text-white shadow-md',
      icon: <WhatsAppIcon />,
      href: 'https://wa.me/12089055973',
    },
  ]

  return (
    <aside
      ref={containerRef}
      className="fixed bottom-6 right-5 sm:bottom-7 sm:right-7 z-50 flex flex-col items-end pointer-events-auto"
      aria-label="Quick Connect & Social Menu"
    >
      {/* Expanded Menu Stack */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex flex-col items-end gap-3.5 mb-3.5"
          >
            {items.map((item, index) => {
              const delay = (items.length - 1 - index) * 0.04
              const content = (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 15, scale: 0.85 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.85 }}
                  transition={{ duration: 0.22, delay }}
                  className="flex items-center justify-end gap-3 group cursor-pointer"
                >
                  {/* Pill label on the left */}
                  <span
                    className={`px-4 py-2 rounded-full text-sm leading-none whitespace-nowrap transition-transform duration-200 group-hover:scale-105 select-none ${item.pillClass}`}
                  >
                    {item.label}
                  </span>

                  {/* Circular icon button on the right */}
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 group-hover:scale-110 active:scale-95 ${item.circleClass}`}
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
                    aria-label={item.label}
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
                  aria-label={item.label}
                  className="focus:outline-none"
                >
                  {content}
                </button>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Trigger Button */}
      <motion.button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close menu' : 'Open quick connect menu'}
        aria-expanded={isOpen}
        whileTap={{ scale: 0.92 }}
        className={`w-14 h-14 rounded-full flex items-center justify-center shadow-xl transition-all duration-300 focus:outline-none cursor-pointer ${
          isOpen
            ? 'bg-[#18181b] text-[#FFBE0B] hover:bg-black'
            : 'bg-[#FFBE0B] text-gray-950 hover:bg-[#f5b300] hover:scale-105'
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
