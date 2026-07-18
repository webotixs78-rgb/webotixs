'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowRight, Play, Star } from 'lucide-react'
import GradientMesh from '@/components/animations/GradientMesh'
import { getCMSData } from '@/lib/data/cms'

const defaultHero = {
  badge: 'Trusted by 200+ Global Clients',
  titlePrefix: 'We Build',
  titleHighlight: 'Digital',
  titleSuffix: 'Experiences',
  subtitle: 'Premium web design, mobile apps, and brand identities crafted for ambitious businesses. We turn your vision into stunning digital products.',
  primaryCtaText: 'Start Your Project',
  primaryCtaLink: '/contact',
  secondaryCtaText: 'View Our Work',
  secondaryCtaLink: '/portfolio',
}

export default function HeroSection() {
  const [hero, setHero] = useState(defaultHero)

  useEffect(() => {
    getCMSData('homepage').then((data) => {
      if (data?.hero) {
        setHero({ ...defaultHero, ...data.hero })
      }
    })
  }, [])

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-background">
      <GradientMesh />

      {/* Grid pattern */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(rgba(248,250,252,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(248,250,252,0.3) 1px, transparent 1px)`,
          backgroundSize: '60px 60px',
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-20">
        <div className="text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-8"
          >
            <div className="flex items-center gap-0.5">
              {[...Array(5)].map((_, i) => (
                <Star key={i} size={10} className="fill-yellow-400 text-yellow-400" />
              ))}
            </div>
            <span className="text-text-gray text-xs font-medium">{hero.badge}</span>
            <span className="w-1.5 h-1.5 bg-success rounded-full animate-pulse" />
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="font-display text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold text-text-white leading-[1.05] tracking-tight mb-6"
          >
            {hero.titlePrefix}{' '}
            <span className="relative inline-block">
              <span className="gradient-text">{hero.titleHighlight}</span>
              <svg
                className="absolute -bottom-2 left-0 w-full"
                viewBox="0 0 300 12"
                fill="none"
              >
                <motion.path
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1, delay: 1 }}
                  d="M2 9 C50 3, 150 3, 298 9"
                  stroke="url(#grad)"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
                <defs>
                  <linearGradient id="grad" x1="0" x2="1" y1="0" y2="0">
                    <stop offset="0%" stopColor="#2563EB" />
                    <stop offset="100%" stopColor="#06B6D4" />
                  </linearGradient>
                </defs>
              </svg>
            </span>{' '}
            <br />
            <span className="text-text-white">{hero.titleSuffix}</span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="text-text-gray text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            {hero.subtitle}
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.55 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
          >
            <Link
              href={hero.primaryCtaLink || '/contact'}
              className="group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all duration-300 hover:scale-105"
            >
              {hero.primaryCtaText}
              <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href={hero.secondaryCtaLink || '/portfolio'}
              className="group flex items-center gap-3 px-8 py-4 glass border border-border rounded-2xl text-text-white font-semibold hover:border-primary/50 hover:text-primary transition-all duration-300"
            >
              <span className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                <Play size={14} className="text-primary ml-0.5" />
              </span>
              {hero.secondaryCtaText}
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl mx-auto"
          >
            {[
              { value: '200+', label: 'Projects Delivered' },
              { value: '98%', label: 'Client Satisfaction' },
              { value: '8+', label: 'Years Experience' },
              { value: '50+', label: 'Team Members' },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.8 + i * 0.1 }}
                className="glass rounded-2xl p-4 border border-border/60"
              >
                <div className="font-display text-3xl font-bold gradient-text mb-1">{stat.value}</div>
                <div className="text-text-gray text-xs">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Floating tech badges */}
        <div className="absolute top-1/4 -left-4 md:left-8 hidden lg:block">
          <motion.div
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            className="glass rounded-2xl px-4 py-3 border border-border/60"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
              <span className="text-xs text-text-gray font-medium">Next.js 16</span>
            </div>
          </motion.div>
        </div>
        <div className="absolute top-1/3 -right-4 md:right-8 hidden lg:block">
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
            className="glass rounded-2xl px-4 py-3 border border-border/60"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              <span className="text-xs text-text-gray font-medium">React 19</span>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  )
}
