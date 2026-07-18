'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, ChevronDown, ArrowRight, Globe, Sun, Moon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTheme } from '@/components/providers/ThemeProvider'

const navigation = [
  { label: 'Home', href: '/' },
  {
    label: 'Services',
    href: '/services',
    children: [
      { label: 'Web Design & Development', href: '/services/web-design-development' },
      { label: 'Mobile App Development', href: '/services/mobile-app-development' },
      { label: 'Brand Identity & Design', href: '/services/brand-identity-design' },
      { label: 'E-Commerce Solutions', href: '/services/ecommerce-solutions' },
      { label: 'SEO & Digital Marketing', href: '/services/seo-digital-marketing' },
      { label: 'Cloud & DevOps', href: '/services/cloud-devops' },
    ],
  },
  { label: 'Portfolio', href: '/portfolio' },
  { label: 'Industries', href: '/industries' },
  { label: 'About', href: '/about' },
  { label: 'Blog', href: '/blog' },
]

const languages = [
  { code: 'EN', name: 'English', dir: 'ltr' },
  { code: 'AR', name: 'العربية', dir: 'rtl' },
  { code: 'ZH', name: '中文', dir: 'ltr' },
  { code: 'UR', name: 'اردو', dir: 'rtl' },
]

export default function Header() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null)
  const [langOpen, setLangOpen] = useState(false)
  const [currentLang, setCurrentLang] = useState('EN')
  const pathname = usePathname()
  const { theme, toggleTheme } = useTheme()

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    setIsMobileOpen(false)
    setActiveDropdown(null)
    setLangOpen(false)
  }, [pathname])

  const handleSelectLang = (code: string, dir: string) => {
    setCurrentLang(code)
    setLangOpen(false)
    document.documentElement.dir = dir
    document.documentElement.lang = code.toLowerCase()
  }

  return (
    <>
      <motion.header
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        suppressHydrationWarning
        className={cn(
          'fixed top-0 left-0 right-0 z-50 transition-all duration-500',
          isScrolled
            ? 'bg-background/80 backdrop-blur-xl border-b border-border/50 shadow-lg'
            : 'bg-transparent'
        )}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2 group">
              <div className="relative w-8 h-8">
                <div className="absolute inset-0 bg-gradient-to-br from-primary-from to-primary-to rounded-lg" />
                <div className="absolute inset-0.5 bg-background rounded-md flex items-center justify-center">
                  <span className="text-xs font-bold gradient-text">W</span>
                </div>
              </div>
              <span className="font-display font-bold text-xl text-text-white group-hover:gradient-text transition-all duration-300">
                Webotixs
              </span>
            </Link>

            {/* Desktop Nav */}
            <nav className="hidden lg:flex items-center gap-1">
              {navigation.map((item) => (
                <div
                  key={item.href}
                  className="relative"
                  onMouseEnter={() => item.children && setActiveDropdown(item.label)}
                  onMouseLeave={() => setActiveDropdown(null)}
                >
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-center gap-1 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200',
                      pathname === item.href
                        ? 'text-primary bg-primary/10'
                        : 'text-text-gray hover:text-text-white hover:bg-white/5'
                    )}
                  >
                    {item.label}
                    {item.children && (
                      <ChevronDown
                        size={14}
                        className={cn(
                          'transition-transform duration-200',
                          activeDropdown === item.label && 'rotate-180'
                        )}
                      />
                    )}
                  </Link>

                  {/* Dropdown */}
                  {item.children && (
                    <AnimatePresence>
                      {activeDropdown === item.label && (
                        <motion.div
                          initial={{ opacity: 0, y: 8, scale: 0.95 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: 8, scale: 0.95 }}
                          transition={{ duration: 0.2 }}
                          className="absolute top-full left-0 mt-2 w-64 glass rounded-2xl p-2 border border-border/60"
                        >
                          {item.children.map((child) => (
                            <Link
                              key={child.href}
                              href={child.href}
                              className="flex items-center justify-between px-3 py-2.5 rounded-xl text-sm text-text-gray hover:text-text-white hover:bg-white/5 transition-all duration-200 group"
                            >
                              <span>{child.label}</span>
                              <ArrowRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                            </Link>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  )}
                </div>
              ))}
            </nav>

            {/* CTA & Language Switcher */}
            <div className="hidden lg:flex items-center gap-3">
              {/* Theme Toggle Button */}
              <button
                onClick={toggleTheme}
                className="flex items-center justify-center w-9 h-9 border border-border/50 rounded-xl text-text-gray hover:text-text-white hover:border-primary/50 bg-background-secondary/50 hover:bg-white/5 transition-all duration-300"
                title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
                aria-label="Toggle Theme"
              >
                {theme === 'dark' ? (
                  <Sun size={16} className="text-amber-400 hover:rotate-45 transition-transform duration-300" />
                ) : (
                  <Moon size={16} className="text-blue-500 hover:-rotate-12 transition-transform duration-300" />
                )}
              </button>

              {/* Language Selector */}
              <div className="relative">
                <button
                  onClick={() => setLangOpen(!langOpen)}
                  className="flex items-center gap-1.5 px-3 py-2 border border-border/50 rounded-xl text-xs font-semibold text-text-gray hover:text-text-white hover:border-primary/50 transition-colors"
                >
                  <Globe size={14} className="text-primary" />
                  <span>{currentLang}</span>
                  <ChevronDown size={12} />
                </button>

                <AnimatePresence>
                  {langOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 6, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 6, scale: 0.95 }}
                      className="absolute right-0 top-full mt-2 w-36 glass rounded-2xl p-1.5 border border-border/60 z-50 shadow-xl"
                    >
                      {languages.map((l) => (
                        <button
                          key={l.code}
                          onClick={() => handleSelectLang(l.code, l.dir)}
                          className={cn(
                            'flex items-center justify-between w-full px-3 py-2 rounded-xl text-xs font-medium transition-colors text-left',
                            currentLang === l.code
                              ? 'bg-primary text-white'
                              : 'text-text-gray hover:text-text-white hover:bg-white/5'
                          )}
                        >
                          <span>{l.name}</span>
                          <span className="text-[10px] opacity-70">{l.code}</span>
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <Link
                href="/contact"
                className="px-5 py-2.5 bg-gradient-to-r from-primary-from to-primary-to text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all duration-300 hover:scale-105"
              >
                Get Started
              </Link>
            </div>

            {/* Mobile menu and theme buttons */}
            <div className="flex items-center gap-2 lg:hidden">
              <button
                onClick={toggleTheme}
                className="p-2 rounded-xl border border-border/50 text-text-gray hover:text-text-white hover:bg-white/5 transition-colors"
                title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
                aria-label="Toggle Theme"
              >
                {theme === 'dark' ? (
                  <Sun size={18} className="text-amber-400" />
                ) : (
                  <Moon size={18} className="text-blue-500" />
                )}
              </button>
              <button
                onClick={() => setIsMobileOpen(!isMobileOpen)}
                className="p-2 rounded-xl text-text-gray hover:text-text-white hover:bg-white/5 transition-colors"
                aria-label="Toggle menu"
              >
                {isMobileOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileOpen && (
          <motion.div
            initial={{ opacity: 0, x: '100%' }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: '100%' }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="fixed inset-0 z-40 bg-background/95 backdrop-blur-xl lg:hidden"
          >
            <div className="flex flex-col h-full pt-24 px-6 pb-8 overflow-y-auto">
              <nav className="flex flex-col gap-1 flex-1">
                {navigation.map((item, i) => (
                  <motion.div
                    key={item.href}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.06 }}
                  >
                    <Link
                      href={item.href}
                      className={cn(
                        'block px-4 py-3 rounded-xl text-lg font-medium transition-all',
                        pathname === item.href
                          ? 'text-primary bg-primary/10'
                          : 'text-text-gray hover:text-text-white hover:bg-white/5'
                      )}
                    >
                      {item.label}
                    </Link>
                    {item.children && (
                      <div className="ml-4 mt-1 flex flex-col gap-0.5">
                        {item.children.map((child) => (
                          <Link
                            key={child.href}
                            href={child.href}
                            className="block px-4 py-2 rounded-xl text-sm text-text-gray hover:text-text-white hover:bg-white/5 transition-all"
                          >
                            {child.label}
                          </Link>
                        ))}
                      </div>
                    )}
                  </motion.div>
                ))}
              </nav>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
                className="mt-4 flex items-center justify-between p-4 bg-background-secondary rounded-2xl border border-border/50"
              >
                <span className="text-sm font-medium text-text-white flex items-center gap-2">
                  {theme === 'dark' ? <Moon size={16} className="text-blue-400" /> : <Sun size={16} className="text-amber-400" />}
                  {theme === 'dark' ? 'Dark Mode' : 'Light Mode'}
                </span>
                <button
                  onClick={toggleTheme}
                  className="px-3 py-1.5 bg-primary/20 text-primary text-xs font-semibold rounded-lg hover:bg-primary/30 transition-colors"
                >
                  Switch to {theme === 'dark' ? 'Light' : 'Dark'}
                </button>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="mt-4"
              >
                <Link
                  href="/contact"
                  className="block w-full py-4 text-center bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl"
                >
                  Get Started
                </Link>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
