'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Calendar, Clock, ArrowRight } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { formatDate, slugify } from '@/lib/utils'
import { getCMSData } from '@/lib/data/cms'

import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export default function BlogGridClient({ initialData }: { initialData: any[] }) {
  const [items, setItems] = useState<any[]>(() => initialData || [])

  useEffect(() => {
    let isMounted = true

    const loadData = () => {
      getCMSData<any[]>('blogs').then((data) => {
        if (!isMounted) return
        if (Array.isArray(data) && data.length > 0) {
          setItems(data)
        }
      })
    }

    loadData()

    const handleStorage = () => {
      try {
        const local = localStorage.getItem('webotixs_cms_blogs')
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
      {items.map((post: any, i: number) => (
        <ScrollReveal key={post.id || i} delay={i * 0.1}>
          <Link href={`/blog/${post.slug || slugify(post.title || '') || post.id}`} className="group block h-full">
            <GlowingGlassCard className="h-full bg-background-card/90 border border-border rounded-3xl overflow-hidden flex flex-col justify-between">
              <div>
                <div className="relative h-48 bg-gradient-to-br from-background-section to-background-secondary overflow-hidden flex items-center justify-center border-b border-border/50">
                  {post.featured_image || post.cover_image || post.image || post.thumbnail ? (
                    <img
                      src={post.featured_image || post.cover_image || post.image || post.thumbnail}
                      alt={post.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/25 to-primary-to/25 flex items-center justify-center border border-primary/20">
                      <span className="font-display font-bold text-2xl gradient-text">{post.title?.[0] || 'B'}</span>
                    </div>
                  )}
                  <div className="absolute top-4 left-4 z-10">
                    <span className="px-2.5 py-1 glass rounded-lg text-[10px] text-text-white border border-border/60 uppercase font-semibold">
                      {post.category?.name || 'Article'}
                    </span>
                  </div>
                </div>

                <div className="p-6">
                  <div className="flex items-center gap-4 text-text-gray text-xs mb-3">
                    <span className="flex items-center gap-1">
                      <Calendar size={12} />
                      {post.published_at ? formatDate(post.published_at) : 'Draft'}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock size={12} />
                      {post.reading_time || 5} min read
                    </span>
                  </div>

                  <div className="font-display text-xl font-bold text-text-white mb-2.5 group-hover:gradient-text transition-colors duration-300">
                    {post.title}
                  </div>
                  <p className="text-text-gray text-sm leading-relaxed mb-4 line-clamp-3">
                    {post.excerpt}
                  </p>
                </div>
              </div>

              <div className="px-6 pb-6">
                <div className="flex items-center gap-2 text-primary text-xs font-semibold group-hover:underline">
                  Read Article
                  <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </GlowingGlassCard>
          </Link>
        </ScrollReveal>
      ))}
    </div>
  )
}
