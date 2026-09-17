'use client'

import { useState, useEffect } from 'react'
import { Star, Quote, Sparkles, X, ChevronRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import ScrollReveal from '@/components/animations/ScrollReveal'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'
import { mockTestimonials } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'
import type { Testimonial } from '@/lib/types'

export default function TestimonialsSection() {
  const [testimonials, setTestimonials] = useState<Testimonial[]>(mockTestimonials)
  const [selectedTestimonial, setSelectedTestimonial] = useState<Testimonial | null>(null)

  useEffect(() => {
    getCMSData<Testimonial[]>('testimonials').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setTestimonials(data)
      }
    })
  }, [])

  // Close modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedTestimonial(null)
    }
    if (selectedTestimonial) {
      window.addEventListener('keydown', handleKeyDown)
    }
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedTestimonial])

  // Distribute testimonials evenly into two rows
  const row1Base: Testimonial[] = []
  const row2Base: Testimonial[] = []

  if (testimonials.length > 0) {
    testimonials.forEach((item, idx) => {
      if (idx % 2 === 0) {
        row1Base.push(item)
      } else {
        row2Base.push(item)
      }
    })

    // If odd or single item, balance so both rows have cards
    if (row2Base.length === 0) {
      row2Base.push(...row1Base)
    } else if (row2Base.length < row1Base.length && row1Base.length > 1) {
      row2Base.push(row1Base[0])
    }
  }

  // Create an extended array duplicated for infinite seamless marquee loop
  const buildTrackList = (baseList: Testimonial[]) => {
    if (baseList.length === 0) return []
    let seq: Testimonial[] = []
    while (seq.length < 4) {
      seq = [...seq, ...baseList]
    }
    // Duplicate sequence for -50% translateX translation loop
    return [...seq, ...seq]
  }

  const row1Items = buildTrackList(row1Base)
  const row2Items = buildTrackList(row2Base)

  const renderCard = (item: Testimonial, key: string) => {
    const isLong = item.content && item.content.length > 125

    return (
      <GlowingGlassCard
        key={key}
        className="w-[340px] sm:w-[420px] md:w-[460px] h-[255px] flex-shrink-0 flex flex-col justify-between glass border border-border/60 rounded-3xl p-6 sm:p-7 hover:border-primary/50 hover:shadow-glow-sm transition-all duration-300 relative group select-none cursor-default"
      >
        <Quote
          size={40}
          className="absolute top-6 right-6 text-primary/10 group-hover:text-primary/25 transition-colors pointer-events-none"
        />

        <div className="relative z-10 flex flex-col justify-between flex-1">
          <div>
            {/* Rating Stars */}
            <div className="flex items-center gap-1.5 mb-3.5">
              {[...Array(item.rating || 5)].map((_, i) => (
                <Star key={i} size={16} className="fill-yellow-400 text-yellow-400" />
              ))}
            </div>

            {/* Clamped Review Content to keep cards compact & uniform */}
            <p className="text-text-white text-sm sm:text-[15px] leading-relaxed font-medium line-clamp-3">
              &ldquo;{item.content}&rdquo;
            </p>

            {/* Read More button if content was hidden / clamped */}
            {isLong && (
              <button
                type="button"
                onClick={() => setSelectedTestimonial(item)}
                className="mt-2 text-xs font-semibold text-primary hover:text-cyan-400 inline-flex items-center gap-1 transition-colors cursor-pointer group-hover:underline"
              >
                Read full review
                <ChevronRight size={13} className="transition-transform group-hover:translate-x-0.5" />
              </button>
            )}
          </div>

          {/* Author Info */}
          <div className="flex items-center gap-3.5 pt-4 border-t border-border/40 mt-auto">
            {item.avatar || (item as any).photo || (item as any).image ? (
              <img
                src={item.avatar || (item as any).photo || (item as any).image}
                alt={item.name}
                className="w-12 h-12 rounded-2xl object-cover border border-primary/30 flex-shrink-0"
              />
            ) : (
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/20 flex-shrink-0">
                <span className="font-display font-bold text-lg gradient-text">
                  {item.name ? item.name[0] : 'W'}
                </span>
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="font-display font-bold text-text-white text-sm sm:text-base truncate">
                {item.name}
              </div>
              <div className="text-text-gray text-xs truncate mt-0.5">
                {item.position}
                {item.company ? `, ${item.company}` : ''}
              </div>
            </div>
          </div>
        </div>
      </GlowingGlassCard>
    )
  }

  return (
    <section className="py-24 bg-background-section relative overflow-hidden border-t border-border/50">
      {/* Background Ambience */}
      <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <ScrollReveal className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-4">
            <Sparkles size={14} className="text-yellow-400" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">
              Client Feedback & Reviews
            </span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-4">
            Loved by Global <span className="gradient-text">Enterprise Leaders</span>
          </h2>
          <p className="text-text-gray text-base sm:text-lg leading-relaxed">
            Explore what CTOs, Founders, and Directors have to say about our engineering quality,
            digital delivery, and technical expertise.
          </p>
        </ScrollReveal>
      </div>

      {/* Auto-sliding 2-Row Slider with Side Fade Masks */}
      <div className="relative w-full overflow-hidden py-4">
        {/* Left & Right Gradient Fade Masks */}
        <div className="absolute left-0 top-0 bottom-0 w-16 sm:w-36 bg-gradient-to-r from-background-section via-background-section/80 to-transparent z-20 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-16 sm:w-36 bg-gradient-to-l from-background-section via-background-section/80 to-transparent z-20 pointer-events-none" />

        <div className="flex flex-col gap-6">
          {/* Row 1: Slides Left */}
          <div className="animate-marquee flex items-stretch gap-6">
            {row1Items.map((item, idx) => renderCard(item, `row1-${item.id || idx}-${idx}`))}
          </div>

          {/* Row 2: Slides Right */}
          <div className="animate-marquee-reverse flex items-stretch gap-6">
            {row2Items.map((item, idx) => renderCard(item, `row2-${item.id || idx}-${idx}`))}
          </div>
        </div>
      </div>

      {/* Full Testimonial Modal (Triggered by "Read full review") */}
      <AnimatePresence>
        {selectedTestimonial && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
            onClick={() => setSelectedTestimonial(null)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ duration: 0.2 }}
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-xl glass border border-primary/40 rounded-3xl p-7 sm:p-9 shadow-2xl overflow-hidden"
            >
              {/* Background ambient lighting */}
              <div className="absolute -top-24 -left-24 w-60 h-60 bg-primary/20 rounded-full blur-3xl pointer-events-none" />
              <div className="absolute -bottom-24 -right-24 w-60 h-60 bg-cyan-500/15 rounded-full blur-3xl pointer-events-none" />

              {/* Close Button */}
              <button
                type="button"
                onClick={() => setSelectedTestimonial(null)}
                aria-label="Close dialog"
                className="absolute top-6 right-6 w-9 h-9 rounded-full glass border border-border/60 flex items-center justify-center text-text-gray hover:text-text-white hover:border-primary/50 transition-colors cursor-pointer z-10"
              >
                <X size={18} />
              </button>

              {/* Modal Content */}
              <div className="relative z-10">
                {/* Rating */}
                <div className="flex items-center gap-1.5 mb-5">
                  {[...Array(selectedTestimonial.rating || 5)].map((_, i) => (
                    <Star key={i} size={18} className="fill-yellow-400 text-yellow-400" />
                  ))}
                </div>

                {/* Full Unclipped Review Content */}
                <div className="max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar mb-6">
                  <p className="text-text-white text-base sm:text-lg leading-relaxed font-medium">
                    &ldquo;{selectedTestimonial.content}&rdquo;
                  </p>
                </div>

                {/* Author Info */}
                <div className="flex items-center gap-4 pt-5 border-t border-border/50">
                  {selectedTestimonial.avatar ||
                  (selectedTestimonial as any).photo ||
                  (selectedTestimonial as any).image ? (
                    <img
                      src={
                        selectedTestimonial.avatar ||
                        (selectedTestimonial as any).photo ||
                        (selectedTestimonial as any).image
                      }
                      alt={selectedTestimonial.name}
                      className="w-14 h-14 rounded-2xl object-cover border border-primary/30 flex-shrink-0"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/20 flex-shrink-0">
                      <span className="font-display font-bold text-xl gradient-text">
                        {selectedTestimonial.name ? selectedTestimonial.name[0] : 'W'}
                      </span>
                    </div>
                  )}
                  <div className="min-w-0">
                    <div className="font-display font-bold text-text-white text-base sm:text-lg truncate">
                      {selectedTestimonial.name}
                    </div>
                    <div className="text-text-gray text-xs sm:text-sm truncate mt-0.5">
                      {selectedTestimonial.position}
                      {selectedTestimonial.company ? `, ${selectedTestimonial.company}` : ''}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </section>
  )
}
