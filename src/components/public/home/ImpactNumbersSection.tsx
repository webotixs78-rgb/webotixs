'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import ScrollReveal from '@/components/animations/ScrollReveal'
import AnimatedCounter from '@/components/animations/AnimatedCounter'
import { Sparkles, Trophy, Users, Globe, Eye } from 'lucide-react'
import { getCMSData } from '@/lib/data/cms'

interface StatItem {
  id: string
  label: string
  value: string
  suffix: string
}

const defaultStats: StatItem[] = [
  { id: '1', label: 'Successful Projects Launched', value: '150', suffix: '+' },
  { id: '2', label: 'Happy Clients Worldwide', value: '98', suffix: '%' },
  { id: '3', label: 'Years of Industry Experience', value: '10', suffix: '+' },
  { id: '4', label: 'Digital Impressions Generated', value: '25', suffix: 'M+' },
]

const icons = [Trophy, Users, Globe, Eye]

import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export default function ImpactNumbersSection() {
  const [stats, setStats] = useState<StatItem[]>(defaultStats)

  useEffect(() => {
    getCMSData('homepage').then((parsed) => {
      if (parsed?.stats && Array.isArray(parsed.stats)) {
        setStats(parsed.stats)
      }
    })
  }, [])

  return (
    <section className="py-20 bg-background-secondary border-y border-border/60 relative overflow-hidden">
      {/* Background radial highlight */}
      <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <ScrollReveal className="text-center mb-14">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-blue-500/30 mb-4">
            <Sparkles size={14} className="text-cyan-400" />
            <span className="text-cyan-400 text-xs font-bold uppercase tracking-wider">Proven Track Record</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white">
            Let’s Discover Our <span className="gradient-text">Impact in Numbers</span>
          </h2>
        </ScrollReveal>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
          {(stats || []).map((item: any, idx: number) => {
            const Icon = icons[idx % icons.length]
            const numericValue = parseInt(String(item.value || '100').replace(/[^0-9]/g, ''), 10) || 100

            return (
              <ScrollReveal key={item.id || idx} delay={idx * 0.1}>
                <GlowingGlassCard className="glass border border-border/60 rounded-3xl p-8 text-center hover:border-primary/50 hover:shadow-glow-md transition-all duration-300 relative group">
                  <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-600/20 to-cyan-500/20 border border-blue-500/30 flex items-center justify-center text-blue-400 mx-auto mb-5 group-hover:scale-110 transition-transform">
                    <Icon size={24} />
                  </div>

                  <div className="font-display text-4xl sm:text-5xl font-extrabold text-white tracking-tight flex items-center justify-center gap-0.5 mb-2">
                    <AnimatedCounter end={numericValue} duration={2.5} />
                    <span className="gradient-text font-bold">{item.suffix}</span>
                  </div>

                  <p className="text-text-gray text-xs sm:text-sm font-semibold uppercase tracking-wider">
                    {item.label}
                  </p>
                </GlowingGlassCard>
              </ScrollReveal>
            )
          })}
        </div>
      </div>
    </section>
  )
}
