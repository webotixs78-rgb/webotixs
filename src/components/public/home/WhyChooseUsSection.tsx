'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ShieldCheck, Cpu, Zap, Globe, Award, HeartHandshake, CheckCircle2 } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'

const defaultReasons = [
  {
    icon: Cpu,
    title: 'High-Concurrency Engineering',
    description: 'We build robust, microservices-ready architectures designed to handle millions of simultaneous transactions with 99.99% uptime guarantees.',
  },
  {
    icon: Zap,
    title: 'Futuristic UI/UX Aesthetics',
    description: 'Our award-winning design team blends immersive micro-animations, glassmorphism, and curated color tokens to captivate users instantly.',
  },
  {
    icon: Globe,
    title: 'Global Delivery & SLAs',
    description: 'With offices across strategic tech hubs and remote teams, we provide round-the-clock enterprise support and strict SLA compliance.',
  },
  {
    icon: ShieldCheck,
    title: 'Zero-Compromise Security',
    description: 'End-to-end encryption, automated OWASP security scanning, and full compliance with GDPR, HIPAA, and UAE data privacy regulations.',
  },
  {
    icon: Award,
    title: 'AI-Powered Growth Strategies',
    description: 'From intelligent lead classification to predictive conversion optimization, we embed cutting-edge AI directly into your digital ecosystem.',
  },
  {
    icon: HeartHandshake,
    title: 'Transparent Agile Roadmap',
    description: 'Weekly sprint demos, shared real-time Notion/Jira boards, and zero hidden costs ensure you are in full control of your digital investment.',
  },
]

import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export default function WhyChooseUsSection() {
  const [reasons, setReasons] = useState<any[]>(defaultReasons)

  useEffect(() => {
    getCMSData('homepage').then((data) => {
      if (Array.isArray(data?.reasons) && data.reasons.length > 0) {
        setReasons(data.reasons.map((r: any, idx: number) => ({
          ...r,
          icon: defaultReasons[idx % defaultReasons.length]?.icon || Cpu,
          description: r.desc || r.description || '',
        })))
      }
    })
  }, [])

  return (
    <section className="py-24 bg-background relative overflow-hidden border-t border-border/50">
      {/* Background glow orbs */}
      <div className="absolute top-1/2 left-0 w-96 h-96 bg-primary-from/10 rounded-full blur-3xl -translate-y-1/2 pointer-events-none" />
      <div className="absolute top-1/2 right-0 w-96 h-96 bg-primary-to/10 rounded-full blur-3xl -translate-y-1/2 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Header */}
        <ScrollReveal className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-4">
            <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse" />
            <span className="text-text-gray text-xs font-semibold uppercase tracking-wider">The Webotixs Advantage</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-5">
            Why Enterprise Leaders Partner with Us to <span className="gradient-text">Build Digital Experiences</span>
          </h2>
          <p className="text-text-gray text-base sm:text-lg leading-relaxed">
            We don’t just build websites — we architect high-performance digital engines engineered to dominate your industry and scale without friction.
          </p>
        </ScrollReveal>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {(reasons || []).map((item: any, idx: number) => {
            const Icon = item.icon || Cpu
            return (
              <ScrollReveal key={item.title || idx} delay={idx * 0.1}>
                <GlowingGlassCard className="group h-full glass border border-border/60 rounded-3xl p-8 hover:border-primary/50 hover:shadow-glow-sm transition-all duration-300 flex flex-col justify-between">
                  <div>
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/20 to-primary-to/20 border border-primary/30 flex items-center justify-center text-primary mb-6 group-hover:scale-110 group-hover:shadow-glow-sm transition-all">
                      <Icon size={26} />
                    </div>
                    <div className="font-display text-xl font-bold text-text-white mb-3 group-hover:text-primary transition-colors">
                      {item.title}
                    </div>
                    <p className="text-text-gray text-sm leading-relaxed mb-6">
                      {item.description}
                    </p>
                  </div>

                  <div className="flex items-center gap-2 text-xs font-semibold text-primary pt-4 border-t border-border/30">
                    <CheckCircle2 size={14} className="text-cyan-400" />
                    <span>Enterprise Verified Standard</span>
                  </div>
                </GlowingGlassCard>
              </ScrollReveal>
            )
          })}
        </div>
      </div>
    </section>
  )
}
