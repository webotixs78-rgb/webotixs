'use client'

import { useState, useEffect } from 'react'
import { Linkedin, Globe, Mail, ChevronLeft, ChevronRight } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockTeam } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export default function TeamPreviewSection() {
  const [members, setMembers] = useState<any[]>(mockTeam)
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    getCMSData('team').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setMembers(data)
      }
    }).catch(() => {})
  }, [])

  const prevSlide = () => {
    setCurrentIndex((prev) => (prev === 0 ? Math.max(0, members.length - 3) : prev - 1))
  }

  const nextSlide = () => {
    setCurrentIndex((prev) => (prev >= members.length - 3 ? 0 : prev + 1))
  }

  return (
    <section className="py-24 bg-background-section relative overflow-hidden border-t border-border/50">
      {/* Ambient background lighting */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-gradient-to-tr from-primary/15 to-cyan-500/15 rounded-full blur-[140px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Header */}
        <ScrollReveal className="text-center mb-16 space-y-3">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-2">
            <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Leadership & Innovation</span>
          </div>

          <h2 className="font-display text-4xl sm:text-5xl md:text-6xl font-bold text-text-white tracking-tight">
            Our <span className="gradient-text">Team</span>
          </h2>
          <p className="text-text-gray text-base sm:text-lg max-w-xl mx-auto font-medium">
            Meet the talented professionals driving innovation and excellence
          </p>
        </ScrollReveal>

        {/* Team Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          {members.slice(currentIndex, currentIndex + 3).map((member: any, i: number) => (
            <ScrollReveal key={member.id || i} delay={i * 0.1}>
              <GlowingGlassCard className="glass border border-border/60 rounded-2xl overflow-hidden text-center flex flex-col justify-between h-full group hover:border-primary/50 transition-all duration-300 shadow-2xl">
                <div>
                  {/* Spotlight Image Container */}
                  <div className="relative w-full aspect-[4/5] bg-background-secondary overflow-hidden flex items-center justify-center">
                    {/* Blue & Cyan Spotlight Ring Background */}
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(59,130,246,0.35)_0%,_rgba(6,182,212,0.2)_50%,_transparent_75%)] pointer-events-none" />

                    {(member.photo || member.avatar || member.image) ? (
                      <img
                        src={member.photo || member.avatar || member.image}
                        alt={member.name}
                        className="w-full h-full object-cover relative z-10 group-hover:scale-105 transition-transform duration-500"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center relative z-10">
                        <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/30">
                          <span className="font-display font-bold text-4xl gradient-text">
                            {(member.name || 'Team Member').split(' ').map((n: string) => n[0]).join('')}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Highly Visible Circular Floating Social Media Buttons */}
                    <div className="absolute bottom-4 left-0 right-0 z-20 flex items-center justify-center gap-3">
                      <a
                        href={member.linkedin || 'https://linkedin.com'}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="w-10 h-10 rounded-full glass border border-border text-text-white hover:text-primary hover:border-primary/60 flex items-center justify-center shadow-xl hover:scale-110 transition-transform"
                        aria-label={`${member.name} LinkedIn profile`}
                      >
                        <Linkedin size={16} />
                        <span className="sr-only">{member.name} LinkedIn Profile</span>
                      </a>
                      <a
                        href={member.github || member.website || '/team'}
                        target={member.github || member.website ? '_blank' : undefined}
                        rel={member.github || member.website ? 'noopener noreferrer' : undefined}
                        className="w-10 h-10 rounded-full glass border border-border text-text-white hover:text-primary hover:border-primary/60 flex items-center justify-center shadow-xl hover:scale-110 transition-transform"
                        aria-label={`${member.name} Portfolio Profile`}
                      >
                        <Globe size={16} />
                        <span className="sr-only">{member.name} Portfolio Profile</span>
                      </a>
                      <a
                        href={`mailto:${member.email || 'info@webotixs.com'}`}
                        className="w-10 h-10 rounded-full glass border border-border text-text-white hover:text-primary hover:border-primary/60 flex items-center justify-center shadow-xl hover:scale-110 transition-transform"
                        aria-label={`Send email to ${member.name}`}
                      >
                        <Mail size={16} />
                        <span className="sr-only">Email {member.name}</span>
                      </a>
                    </div>
                  </div>

                  {/* Card Info */}
                  <div className="p-6 space-y-2">
                    <div className="font-display font-bold text-2xl text-text-white group-hover:gradient-text transition-colors">
                      {member.name}
                    </div>
                    <p className="text-primary text-xs font-semibold uppercase tracking-wider">
                      {member.position}
                    </p>
                    <p className="text-text-gray text-xs leading-relaxed pt-2">
                      {member.bio}
                    </p>
                  </div>
                </div>
              </GlowingGlassCard>
            </ScrollReveal>
          ))}
        </div>

        {/* Carousel Navigation Buttons */}
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={prevSlide}
            className="w-12 h-12 rounded-full glass border border-border/80 hover:border-primary/50 text-text-white hover:text-primary flex items-center justify-center shadow-lg transition-all hover:scale-105 active:scale-95"
            aria-label="Previous team slide"
          >
            <ChevronLeft size={20} />
          </button>
          <button
            onClick={nextSlide}
            className="w-12 h-12 rounded-full bg-gradient-to-r from-primary-from to-primary-to text-white flex items-center justify-center shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-105 active:scale-95"
            aria-label="Next team slide"
          >
            <ChevronRight size={20} />
          </button>
        </div>
      </div>
    </section>
  )
}
