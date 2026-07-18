'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, Github, Linkedin, Mail } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockTeam } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'

export default function TeamPreviewSection() {
  const [members, setMembers] = useState<any[]>(mockTeam.filter((m) => m.featured))

  useEffect(() => {
    getCMSData('team').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setMembers(data.filter((m: any) => m.featured).slice(0, 4))
      }
    })
  }, [])

  return (
    <section className="section-padding bg-background relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
            <span className="w-1.5 h-1.5 bg-primary rounded-full" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Our Team</span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-bold text-text-white mb-5">
            Meet the{' '}
            <span className="gradient-text">Experts</span>
          </h2>
          <p className="text-text-gray text-lg max-w-2xl mx-auto">
            A passionate team of designers, developers, and strategists committed to delivering excellence.
          </p>
        </ScrollReveal>

        {/* Team Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {(members || []).map((member: any, i: number) => (
            <ScrollReveal key={member.id || i} delay={i * 0.1}>
              <div className="group bg-background-card rounded-3xl p-6 border border-border card-hover text-center">
                {/* Avatar */}
                <div className="relative w-20 h-20 mx-auto mb-4">
                  {member.photo ? (
                    <img src={member.photo} alt={member.name} className="w-20 h-20 rounded-2xl object-cover border border-primary/20 group-hover:border-primary/40 group-hover:shadow-glow-sm transition-all duration-300" />
                  ) : (
                    <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/20 group-hover:border-primary/40 group-hover:shadow-glow-sm transition-all duration-300">
                      <span className="font-display font-bold text-2xl gradient-text">
                        {(member.name || 'Team Member').split(' ').map((n: string) => n[0]).join('')}
                      </span>
                    </div>
                  )}
                  <div className="absolute -bottom-1 -right-1 w-5 h-5 bg-success rounded-full border-2 border-background-card" />
                </div>

                <h3 className="font-display font-bold text-text-white mb-1 group-hover:gradient-text transition-all duration-300">
                  {member.name}
                </h3>
                <p className="text-primary text-sm mb-3">{member.position}</p>
                <p className="text-text-gray text-xs leading-relaxed mb-5 line-clamp-2">{member.bio}</p>

                {/* Social */}
                <div className="flex items-center justify-center gap-2">
                  {member.linkedin && (
                    <a href={member.linkedin} className="w-8 h-8 glass rounded-lg flex items-center justify-center text-text-gray hover:text-primary transition-colors border border-border">
                      <Linkedin size={13} />
                    </a>
                  )}
                  {member.github && (
                    <a href={member.github} className="w-8 h-8 glass rounded-lg flex items-center justify-center text-text-gray hover:text-primary transition-colors border border-border">
                      <Github size={13} />
                    </a>
                  )}
                  {member.email && (
                    <a href={`mailto:${member.email}`} className="w-8 h-8 glass rounded-lg flex items-center justify-center text-text-gray hover:text-primary transition-colors border border-border">
                      <Mail size={13} />
                    </a>
                  )}
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>

        {/* CTA */}
        <ScrollReveal className="text-center mt-12">
          <Link
            href="/team"
            className="inline-flex items-center gap-2 px-8 py-4 glass border border-border text-text-white font-semibold rounded-2xl hover:border-primary/50 hover:text-primary transition-all duration-300"
          >
            Meet The Full Team <ArrowRight size={16} />
          </Link>
        </ScrollReveal>
      </div>
    </section>
  )
}
