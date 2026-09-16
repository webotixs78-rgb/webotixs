'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Landmark, ShoppingBag, ShieldCheck, GraduationCap, Truck, HeartPulse, Globe } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Landmark,
  ShoppingBag,
  HeartPulse,
  GraduationCap,
  Truck,
  ShieldCheck,
  Globe,
}

import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export default function IndustriesGridClient({ initialData }: { initialData: any[] }) {
  const [items, setItems] = useState<any[]>(() => initialData || [])

  useEffect(() => {
    let isMounted = true

    const loadData = () => {
      getCMSData<any[]>('industries').then((data) => {
        if (!isMounted) return
        if (Array.isArray(data) && data.length > 0) {
          setItems(data)
        }
      })
    }

    loadData()

    const handleStorage = () => {
      try {
        const local = localStorage.getItem('webotixs_cms_industries')
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
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      {items.map((ind: any, i: number) => {
        const Icon = iconMap[ind.icon] || Landmark
        return (
          <ScrollReveal key={ind.id || ind.title || i} delay={i * 0.1}>
            <GlowingGlassCard className="bg-background-card/90 border border-border/60 rounded-3xl p-8 hover:border-primary/40 transition-all duration-300 h-full flex flex-col justify-between overflow-hidden relative">
              <div>
                {(ind.image || ind.cover_image || ind.thumbnail) && (
                  <div className="h-44 -mx-8 -mt-8 mb-6 overflow-hidden border-b border-border/50">
                    <img
                      src={ind.image || ind.cover_image || ind.thumbnail}
                      alt={ind.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                    />
                  </div>
                )}
                <div className="w-14 h-14 bg-primary/10 rounded-2xl flex items-center justify-center mb-6 text-primary border border-primary/20">
                  <Icon size={24} />
                </div>
                <div className="font-display text-2xl font-bold text-text-white mb-3">{ind.title}</div>
                <p className="text-text-gray text-sm leading-relaxed mb-6">{ind.description}</p>

                <div className="space-y-2 mb-8">
                  <div className="text-xs uppercase text-text-white font-bold tracking-wider mb-3">Key Solutions:</div>
                  {(ind.benefits || []).map((benefit: string, j: number) => (
                    <div key={j} className="flex items-center gap-2.5 text-text-gray text-xs">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                      {benefit}
                    </div>
                  ))}
                </div>
              </div>

              <Link
                href="/contact"
                className="inline-flex items-center justify-center gap-2 w-full py-3.5 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-102"
              >
                Discuss Your Industry Needs
              </Link>
            </GlowingGlassCard>
          </ScrollReveal>
        )
      })}
    </div>
  )
}
