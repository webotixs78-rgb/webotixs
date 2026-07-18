'use client'

import { useState, useEffect, useTransition } from 'react'
import { Search, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, FileText, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import { mockServices, mockPortfolio, mockBlogPosts } from '@/lib/data/mock'
import { slugify } from '@/lib/utils'
import ScrollReveal from '@/components/animations/ScrollReveal'

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Globe,
  Smartphone,
  Palette,
  ShoppingCart,
  TrendingUp,
  Cloud,
}

interface SearchResult {
  id: string
  title: string
  url: string
  type: 'Service' | 'Project' | 'Blog'
  desc: string
  icon?: string
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [, startTransition] = useTransition()

  // Pre-index all elements for search
  const dataset: SearchResult[] = [
    ...mockServices.map((s) => ({
      id: s.id,
      title: s.title,
      url: `/services/${s.slug}`,
      type: 'Service' as const,
      desc: s.short_description,
      icon: s.icon,
    })),
    ...mockPortfolio.map((p) => ({
      id: p.id,
      title: p.title,
      url: `/portfolio/${p.id}`,
      type: 'Project' as const,
      desc: p.description,
    })),
    ...mockBlogPosts.map((b) => ({
      id: b.id,
      title: b.title,
      url: `/blog/${b.slug}`,
      type: 'Blog' as const,
      desc: b.excerpt,
    })),
  ]

  useEffect(() => {
    if (!query.trim()) {
      setResults([])
      return
    }

    startTransition(() => {
      const q = query.toLowerCase()
      const filtered = dataset.filter(
        (item) =>
          item.title.toLowerCase().includes(q) ||
          item.desc.toLowerCase().includes(q) ||
          item.type.toLowerCase().includes(q)
      )
      setResults(filtered)
    })
  }, [query])

  return (
    <div className="pt-24 bg-background min-h-screen">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <ScrollReveal className="text-center mb-10">
          <h1 className="font-display text-4xl md:text-5xl font-bold text-text-white mb-4">
            Search Our <span className="gradient-text">Platform</span>
          </h1>
          <p className="text-text-gray text-sm md:text-base max-w-sm mx-auto">
            Find services, active projects, tutorials, or journal logs.
          </p>
        </ScrollReveal>

        {/* Input box */}
        <ScrollReveal className="relative max-w-2xl mx-auto mb-12">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your search query (e.g. Next.js, Design, Commerce)..."
            className="w-full px-6 py-4 bg-background-card border border-border rounded-2xl text-text-white placeholder:text-text-gray/40 focus:outline-none focus:border-primary/50 transition-colors pl-14"
          />
          <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-text-gray/50" size={20} />
        </ScrollReveal>

        {/* Results Matrix */}
        <ScrollReveal className="space-y-4 max-w-2xl mx-auto">
          {results.length > 0 ? (
            results.map((res) => {
              const Icon = res.icon && iconMap[res.icon] ? iconMap[res.icon] : FileText
              return (
                <Link key={res.url} href={res.url} className="group block">
                  <div className="glass border border-border/60 hover:border-primary/40 rounded-2xl p-5 flex items-start gap-4 transition-colors">
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                      <Icon size={18} />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-1">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-primary">
                          {res.type}
                        </span>
                        <h2 className="font-display text-base font-bold text-text-white group-hover:gradient-text transition-colors truncate">
                          {res.title}
                        </h2>
                      </div>
                      <p className="text-text-gray text-xs leading-relaxed line-clamp-2">
                        {res.desc}
                      </p>
                    </div>

                    <ArrowRight size={14} className="text-text-gray/40 group-hover:text-primary group-hover:translate-x-1 transition-all mt-1" />
                  </div>
                </Link>
              )
            })
          ) : query.trim() ? (
            <div className="text-center py-12 text-text-gray text-sm">
              No results found for &ldquo;{query}&rdquo;.
            </div>
          ) : (
            <div className="text-center py-12 text-text-gray/40 text-xs">
              Type above to start searching.
            </div>
          )}
        </ScrollReveal>
      </div>
    </div>
  )
}
