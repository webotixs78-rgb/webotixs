'use client'

import { useState, useEffect } from 'react'
import { Star, Quote, Sparkles } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'
import { mockTestimonials } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'
import type { Testimonial } from '@/lib/types'

export default function TestimonialsSection() {
  const [testimonials, setTestimonials] = useState<Testimonial[]>(mockTestimonials)

  useEffect(() => {
    getCMSData<Testimonial[]>('testimonials').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setTestimonials(data)
      }
    })
  }, [])

  return (
    <section className="py-24 bg-background-section relative overflow-hidden border-t border-border/50">
      <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <ScrollReveal className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-4">
            <Sparkles size={14} className="text-yellow-400" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Client Feedback & Reviews</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-4">
            Loved by Global <span className="gradient-text">Enterprise Leaders</span>
          </h2>
          <p className="text-text-gray text-base sm:text-lg leading-relaxed">
            Explore what CTOs, Founders, and Directors have to say about our engineering quality, digital delivery, and technical expertise.
          </p>
        </ScrollReveal>

        {/* Testimonials Grid - Each Review Rendered Once (Zero Duplicate Content) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {testimonials.map((item: any, idx: number) => (
            <ScrollReveal key={item.id || idx} delay={idx * 0.1}>
              <GlowingGlassCard className="h-full glass border border-border/60 rounded-3xl p-8 sm:p-10 flex flex-col justify-between hover:border-primary/50 hover:shadow-glow-sm transition-all duration-300 relative group">
                <Quote size={44} className="absolute top-8 right-8 text-primary/10 group-hover:text-primary/20 transition-colors pointer-events-none" />

                <div>
                  {/* Rating Stars */}
                  <div className="flex items-center gap-1.5 mb-5">
                    {[...Array(item.rating || 5)].map((_, i) => (
                      <Star key={i} size={18} className="fill-yellow-400 text-yellow-400" />
                    ))}
                  </div>

                  {/* Review Text */}
                  <p className="text-text-white text-base sm:text-lg leading-relaxed font-medium mb-8">
                    &ldquo;{item.content}&rdquo;
                  </p>
                </div>

                {/* Author Info */}
                <div className="flex items-center gap-4 pt-5 border-t border-border/40">
                  {(item.avatar || item.photo || item.image) ? (
                    <img
                      src={item.avatar || item.photo || item.image}
                      alt={item.name}
                      className="w-14 h-14 rounded-2xl object-cover border border-primary/30"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/20 flex-shrink-0">
                      <span className="font-display font-bold text-xl gradient-text">
                        {item.name[0]}
                      </span>
                    </div>
                  )}
                  <div className="min-w-0">
                    <div className="font-display font-bold text-text-white text-base truncate">
                      {item.name}
                    </div>
                    <div className="text-text-gray text-xs truncate mt-0.5">
                      {item.position}{item.company ? `, ${item.company}` : ''}
                    </div>
                  </div>
                </div>
              </GlowingGlassCard>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}
