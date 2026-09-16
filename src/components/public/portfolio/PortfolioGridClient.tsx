'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ExternalLink, ArrowRight, TrendingUp } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { slugify } from '@/lib/utils'
import { getCMSData } from '@/lib/data/cms'

export default function PortfolioGridClient({ initialData }: { initialData: any[] }) {
  const [items, setItems] = useState<any[]>(() => initialData || [])

  useEffect(() => {
    let isMounted = true

    const loadData = () => {
      getCMSData<any[]>('portfolio').then((data) => {
        if (!isMounted) return
        if (Array.isArray(data) && data.length > 0) {
          setItems(data)
        }
      })
    }

    loadData()

    const handleStorage = () => {
      try {
        const local = localStorage.getItem('webotixs_cms_portfolio')
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
      {items.map((project: any, i: number) => (
        <ScrollReveal key={project.id || i} delay={i * 0.1}>
          <Link href={`/portfolio/${project.slug || slugify(project.title || '') || project.id}`} className="group block h-full">
            <div className="portfolio-card">
              <div>
                {/* Image Section */}
                <div className="relative h-60 w-full bg-gradient-to-br from-[#050816] to-[#0A0E1F] overflow-hidden">
                  {project.thumbnail || project.cover_image || project.image ? (
                    <img
                      src={project.thumbnail || project.cover_image || project.image}
                      alt={project.title}
                      className="portfolio-card-img"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <div className="w-16 h-16 rounded-2xl bg-blue-500/15 backdrop-blur-md flex items-center justify-center mx-auto mb-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.15)]">
                          <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                            {project.title?.[0] || 'P'}
                          </span>
                        </div>
                        <span className="text-[#94A3B8]/60 text-xs">{project.client}</span>
                      </div>
                    </div>
                  )}

                  {/* Gradient Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-[#0A0E1F] via-black/20 to-transparent opacity-85 group-hover:opacity-60 transition-opacity duration-300" />

                  {/* Category Tag Badge (Borderless Glass Pill) */}
                  <div className="absolute bottom-4 left-4 z-10">
                    <span className="portfolio-category-pill">
                      {project.industry || 'Web App'}
                    </span>
                  </div>

                  {/* Live URL Badge */}
                  {project.live_url && (
                    <div className="portfolio-ext-link">
                      <ExternalLink size={15} className="text-white" />
                    </div>
                  )}
                </div>

                {/* Card Content */}
                <div className="p-7 space-y-4">
                  <div className="font-display text-xl font-bold text-white group-hover:text-cyan-300 transition-colors duration-300 leading-snug">
                    {project.title}
                  </div>
                  <p className="text-[#94A3B8] text-sm leading-relaxed line-clamp-3">
                    {project.description}
                  </p>

                  {/* Statistics Notification Chips (Borderless Glass) */}
                  <div className="space-y-2 pt-1">
                    {(project.results || []).slice(0, 2).map((res: string, j: number) => (
                      <div key={j} className="portfolio-result-chip">
                        <TrendingUp size={14} className="text-cyan-400 shrink-0" />
                        <span className="truncate">{res}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Card Footer */}
              <div className="px-7 pb-7 pt-2">
                {/* Tech Chips */}
                <div className="flex flex-wrap gap-2 mb-6">
                  {(project.technologies || []).slice(0, 4).map((tech: string, idx: number) => (
                    <span key={idx} className="portfolio-tech-tag">
                      {tech}
                    </span>
                  ))}
                </div>

                <div className="flex items-center gap-2 text-cyan-400 text-sm font-semibold group-hover:underline">
                  View Project Breakdown
                  <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </div>
          </Link>
        </ScrollReveal>
      ))}
    </div>
  )
}
