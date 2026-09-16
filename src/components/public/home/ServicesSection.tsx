'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, Check } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockServices } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'
import { slugify } from '@/lib/utils'

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Globe,
  Smartphone,
  Palette,
  ShoppingCart,
  TrendingUp,
  Cloud,
}

import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export default function ServicesSection() {
  const [services, setServices] = useState<any[]>(mockServices.slice(0, 6))

  useEffect(() => {
    let mounted = true
    getCMSData('services')
      .then((data) => {
        if (mounted && data && Array.isArray(data) && data.length > 0) {
          setServices(data.slice(0, 6))
        }
      })
      .catch(() => {})

    const handleStorage = () => {
      try {
        const local = localStorage.getItem('webotixs_cms_services')
        if (local) {
          const parsed = JSON.parse(local)
          if (Array.isArray(parsed) && parsed.length > 0) {
            setServices(parsed.slice(0, 6))
          }
        }
      } catch {}
    }

    handleStorage()
    window.addEventListener('storage', handleStorage)
    return () => {
      mounted = false
      window.removeEventListener('storage', handleStorage)
    }
  }, [])

  return (
    <section className="py-24 bg-background-secondary border-t border-border/50 relative overflow-hidden">
      <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <ScrollReveal className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-4">
            <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">What We Do</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-4">
            End-to-End <span className="gradient-text">Digital Capabilities</span>
          </h2>
          <p className="text-text-gray text-lg max-w-2xl mx-auto">
            Transformative digital experiences built for scale — from concept to launch, we deliver end-to-end solutions that turn visitors into loyal customers.
          </p>
        </ScrollReveal>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {services.map((service: any, i: number) => {
            const Icon = iconMap[service.icon] ?? Globe
            const serviceHref = `/services/${service.slug || slugify(service.title || '') || service.id}`
            return (
              <ScrollReveal key={service.id || i} delay={i * 0.1}>
                <GlowingGlassCard className="h-full bg-background-card/90 rounded-3xl p-8 border border-border/60 flex flex-col justify-between overflow-hidden group">
                  <div>
                    {(service.cover_image || service.image || service.thumbnail) && (
                      <Link
                        href={serviceHref}
                        aria-label={`View ${service.title}`}
                        className="block h-44 -mx-8 -mt-8 mb-6 overflow-hidden border-b border-border/50"
                      >
                        <img
                          src={service.cover_image || service.image || service.thumbnail}
                          alt={service.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                      </Link>
                    )}
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

                    <Link
                      href={serviceHref}
                      className="font-display text-xl font-bold text-text-white mb-3 hover:text-primary transition-colors duration-300 block"
                    >
                      {service.title}
                    </Link>
                    <p className="text-text-gray text-sm leading-relaxed mb-6 line-clamp-3">
                      {service.short_description || service.long_description || service.description}
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
                    <Link
                      href={serviceHref}
                      aria-label={`Explore ${service.title} services`}
                      className="inline-flex items-center gap-2 text-primary text-sm font-semibold hover:underline"
                    >
                      <span>Explore {service.title}</span>
                      <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                    </Link>
                  </div>
                </GlowingGlassCard>
              </ScrollReveal>
            )
          })}
        </div>

        {/* CTA */}
        <ScrollReveal className="text-center mt-12">
          <Link
            href="/services"
            aria-label="Explore all Webotixs agency services"
            className="btn-float-rtl-glass inline-flex items-center gap-2 px-8 py-4 glass border border-border text-text-white font-bold rounded-2xl"
          >
            View All Services <ArrowRight size={16} />
          </Link>
        </ScrollReveal>
      </div>
    </section>
  )
}
