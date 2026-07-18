'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, Check } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockServices } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Globe,
  Smartphone,
  Palette,
  ShoppingCart,
  TrendingUp,
  Cloud,
}

export default function ServicesSection() {
  const [services, setServices] = useState<any[]>(mockServices.slice(0, 6))

  useEffect(() => {
    getCMSData('services').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setServices(data.slice(0, 6))
      }
    })
  }, [])

  return (
    <section className="section-padding bg-background-secondary relative overflow-hidden">
      <div className="absolute inset-0 mesh-gradient opacity-30" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
            <span className="w-1.5 h-1.5 bg-primary rounded-full" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">What We Do</span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-bold text-text-white mb-5">
            Services Built for{' '}
            <span className="gradient-text">Growth</span>
          </h2>
          <p className="text-text-gray text-lg max-w-2xl mx-auto">
            From concept to launch, we deliver end-to-end digital solutions that drive measurable results for your business.
          </p>
        </ScrollReveal>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {services.map((service: any, i: number) => {
            const Icon = iconMap[service.icon] ?? Globe
            return (
              <ScrollReveal key={service.id || i} delay={i * 0.1}>
                <div className="group h-full bg-background-card rounded-3xl p-8 border border-border/60 card-hover flex flex-col justify-between">
                  <div>
                    {/* Icon & badge */}
                    <div className="flex items-center justify-between mb-6">
                      <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/20 to-primary-to/20 flex items-center justify-center border border-primary/20 group-hover:border-primary/50 group-hover:shadow-glow-sm transition-all duration-300">
                        <Icon size={24} className="text-primary" />
                      </div>
                      {service.featured && (
                        <span className="px-3 py-1 glass rounded-full text-[11px] text-primary border border-primary/25 font-semibold">
                          Featured
                        </span>
                      )}
                    </div>

                    <h3 className="font-display text-xl font-bold text-text-white mb-3 group-hover:gradient-text transition-colors duration-300">
                      {service.title}
                    </h3>
                    <p className="text-text-gray text-sm leading-relaxed mb-6 line-clamp-3">
                      {service.short_description || service.long_description}
                    </p>

                    {/* Features list */}
                    <ul className="space-y-2.5 mb-8">
                      {(service.features || []).slice(0, 3).map((feat: string, idx: number) => (
                        <li key={idx} className="flex items-center gap-2.5 text-text-gray text-sm">
                          <span className="w-5 h-5 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                            <Check size={12} />
                          </span>
                          {feat}
                        </li>
                      ))}
                    </ul>
                    <div className="flex items-center gap-2 text-primary text-sm font-medium">
                      Learn More
                      <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            )
          })}
        </div>

        {/* CTA */}
        <ScrollReveal className="text-center mt-12">
          <Link
            href="/services"
            className="inline-flex items-center gap-2 px-8 py-4 glass border border-border text-text-white font-semibold rounded-2xl hover:border-primary/50 hover:text-primary transition-all duration-300"
          >
            View All Services <ArrowRight size={16} />
          </Link>
        </ScrollReveal>
      </div>
    </section>
  )
}
