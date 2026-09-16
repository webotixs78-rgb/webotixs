'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Check, ArrowRight, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, Database, Cpu } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'
import { slugify } from '@/lib/utils'
import { getCMSData } from '@/lib/data/cms'

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Globe,
  Smartphone,
  Palette,
  ShoppingCart,
  TrendingUp,
  Cloud,
  Database,
  Cpu,
}

export default function ServicesGridClient({ initialData }: { initialData: any[] }) {
  const [items, setItems] = useState<any[]>(() => initialData || [])

  useEffect(() => {
    let isMounted = true

    const loadData = () => {
      getCMSData<any[]>('services').then((data) => {
        if (!isMounted) return
        if (Array.isArray(data) && data.length > 0) {
          setItems(data)
        }
      })
    }

    loadData()

    const handleStorage = () => {
      try {
        const local = localStorage.getItem('webotixs_cms_services')
        if (local) {
          const parsed = JSON.parse(local)
          if (Array.isArray(parsed) && parsed.length > 0) {
            setItems(parsed)
          }
        }
      } catch {}
    }

    window.addEventListener('storage', handleStorage)
    return () => {
      isMounted = false
      window.removeEventListener('storage', handleStorage)
    }
  }, [])

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
      {items.map((service: any, i: number) => {
        const Icon = iconMap[service.icon] ?? Globe
        return (
          <ScrollReveal key={service.id || i} delay={i * 0.1}>
            <GlowingGlassCard className="h-full bg-background-card/90 border border-border/60 rounded-3xl p-8 flex flex-col justify-between overflow-hidden relative">
              <div>
                {(service.cover_image || service.image || service.thumbnail) && (
                  <div className="h-44 -mx-8 -mt-8 mb-6 overflow-hidden border-b border-border/50">
                    <img
                      src={service.cover_image || service.image || service.thumbnail}
                      alt={service.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                    />
                  </div>
                )}
                <div className="w-14 h-14 bg-gradient-to-br from-primary-from/20 to-primary-to/20 rounded-2xl flex items-center justify-center mb-6 border border-primary/20">
                  <Icon size={24} className="text-primary" />
                </div>

                <div className="font-display text-2xl font-bold text-text-white mb-3 hover:gradient-text transition-colors">
                  {service.title}
                </div>
                <p className="text-text-gray text-sm leading-relaxed mb-6">
                  {service.long_description || service.description}
                </p>

                <ul className="space-y-2.5 mb-8">
                  {(service.features || []).map((feat: string, idx: number) => (
                    <li key={idx} className="flex items-center gap-2.5 text-text-gray text-sm">
                      <span className="w-5 h-5 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                        <Check size={12} />
                      </span>
                      {feat}
                    </li>
                  ))}
                </ul>
              </div>

              <Link
                href={`/services/${service.slug || slugify(service.title || '') || service.id}`}
                className="inline-flex items-center justify-center gap-2 w-full py-3.5 glass border border-border hover:border-primary/50 text-text-white hover:text-primary font-semibold rounded-2xl transition-all duration-300 group"
              >
                Explore Service Detail
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
              </Link>
            </GlowingGlassCard>
          </ScrollReveal>
        )
      })}
    </div>
  )
}
