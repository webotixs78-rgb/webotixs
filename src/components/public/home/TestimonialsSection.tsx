'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Star, Quote, Sparkles } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockTestimonials } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'
import type { Testimonial } from '@/lib/types'
import { cn } from '@/lib/utils'

export default function TestimonialsSection() {
  const [testimonials, setTestimonials] = useState<Testimonial[]>(mockTestimonials)

  useEffect(() => {
    getCMSData<Testimonial[]>('testimonials').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setTestimonials(data)
      }
    })
  }, [])

  // We duplicate items so the marquee ticker loops smoothly without gaps
  const row1 = [...testimonials, ...testimonials, ...testimonials]
  const row2 = [...testimonials.slice().reverse(), ...testimonials.slice().reverse(), ...testimonials.slice().reverse()]

  return (
    <section className="py-24 bg-background-section relative overflow-hidden border-t border-border/50">
      <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 mb-16">
        <ScrollReveal className="text-center max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-4">
            <Sparkles size={14} className="text-yellow-400" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Client Feedback & Reviews</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-4">
            Loved by Global <span className="gradient-text">Enterprise Leaders</span>
          </h2>
          <p className="text-text-gray text-base sm:text-lg">
            Explore what CTOs, Founders, and Directors have to say about our engineering quality and digital delivery.
          </p>
        </ScrollReveal>
      </div>

      {/* Marquee Container */}
      <div className="space-y-8 relative z-10 overflow-hidden py-4">
        {/* Top Row: Slider moving Right to Left */}
        <div className="flex overflow-hidden">
          <motion.div
            className="flex gap-6 flex-shrink-0"
            animate={{ x: ['0%', '-33.333%'] }}
            transition={{
              duration: 35,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            {(row1 || []).map((item: any, idx: number) => (
              <TestimonialCard key={`${item.id || 't'}-row1-${idx}`} item={item} />
            ))}
          </motion.div>
        </div>

        {/* Bottom Row: Slider moving Left to Right */}
        <div className="flex overflow-hidden">
          <motion.div
            className="flex gap-6 flex-shrink-0"
            animate={{ x: ['-33.333%', '0%'] }}
            transition={{
              duration: 38,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            {(row2 || []).map((item: any, idx: number) => (
              <TestimonialCard key={`${item.id || 't'}-row2-${idx}`} item={item} />
            ))}
          </motion.div>
        </div>
      </div>

      {/* Bottom fade edges */}
      <div className="absolute top-0 left-0 bottom-0 w-24 bg-gradient-to-r from-background-section to-transparent z-20 pointer-events-none" />
      <div className="absolute top-0 right-0 bottom-0 w-24 bg-gradient-to-l from-background-section to-transparent z-20 pointer-events-none" />
    </section>
  )
}

function TestimonialCard({ item }: { item: any }) {
  return (
    <div className="w-[360px] sm:w-[420px] glass border border-border/60 rounded-3xl p-6 sm:p-8 flex flex-col justify-between hover:border-primary/50 hover:shadow-glow-sm transition-all duration-300 flex-shrink-0 relative group">
      <Quote size={40} className="absolute top-6 right-6 text-primary/10 group-hover:text-primary/20 transition-colors" />

      <div>
        {/* Stars */}
        <div className="flex items-center gap-1 mb-4">
          {[...Array(item.rating || 5)].map((_, i) => (
            <Star key={i} size={16} className="fill-yellow-400 text-yellow-400" />
          ))}
        </div>

        {/* Content */}
        <p className="text-text-white text-sm sm:text-base leading-relaxed font-medium mb-6 line-clamp-4">
          &ldquo;{item.content}&rdquo;
        </p>
      </div>

      {/* Author Info */}
      <div className="flex items-center gap-3 pt-4 border-t border-border/40">
        {item.avatar ? (
          <img src={item.avatar} alt={item.name} className="w-12 h-12 rounded-2xl object-cover border border-primary/30" />
        ) : (
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/20 flex-shrink-0">
            <span className="font-display font-bold text-lg gradient-text">
              {item.name[0]}
            </span>
          </div>
        )}
        <div className="min-w-0">
          <div className="font-display font-bold text-text-white text-sm truncate">{item.name}</div>
          <div className="text-text-gray text-xs truncate">
            {item.position}{item.company ? `, ${item.company}` : ''}
          </div>
        </div>
      </div>
    </div>
  )
}
