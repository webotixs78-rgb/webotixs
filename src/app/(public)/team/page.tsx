import { Github, Linkedin, Mail, Globe } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Our Team',
  description: 'Meet the brilliant minds behind Webotixs. Designers, developers, and digital marketing leaders.',
}

export default async function TeamPage() {
  const team: any[] = await getCMSData('team')

  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">The Innovators</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Our <span className="gradient-text">Team</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              Meet the talented professionals driving innovation and excellence at Webotixs.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Team Matrix */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {team.map((member, i) => (
              <ScrollReveal key={member.id || i} delay={i * 0.1}>
                <GlowingGlassCard className="glass border border-border/60 rounded-2xl overflow-hidden text-center flex flex-col justify-between h-full group hover:border-primary/50 transition-all duration-300 shadow-2xl">
                  <div>
                    {/* Spotlight Image Container */}
                    <div className="relative w-full aspect-[4/5] bg-background-secondary overflow-hidden flex items-center justify-center">
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

                      {/* Floating Social Media Buttons */}
                      <div className="absolute bottom-4 left-0 right-0 z-20 flex items-center justify-center gap-3">
                        {member.linkedin && (
                          <a
                            href={member.linkedin}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="w-10 h-10 rounded-full glass border border-border text-text-white hover:text-primary flex items-center justify-center shadow-lg hover:scale-110 transition-transform"
                            aria-label="LinkedIn profile"
                          >
                            <Linkedin size={16} />
                          </a>
                        )}
                        {(member.github || member.website) && (
                          <a
                            href={member.github || member.website || '#'}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="w-10 h-10 rounded-full glass border border-border text-text-white hover:text-primary flex items-center justify-center shadow-lg hover:scale-110 transition-transform"
                            aria-label="Website"
                          >
                            <Globe size={16} />
                          </a>
                        )}
                        {member.email && (
                          <a
                            href={`mailto:${member.email}`}
                            className="w-10 h-10 rounded-full glass border border-border text-text-white hover:text-primary flex items-center justify-center shadow-lg hover:scale-110 transition-transform"
                            aria-label="Send email"
                          >
                            <Mail size={16} />
                          </a>
                        )}
                      </div>
                    </div>

                    {/* Card Info */}
                    <div className="p-6 space-y-2">
                      <h2 className="font-display font-bold text-2xl text-text-white group-hover:gradient-text transition-colors">
                        {member.name}
                      </h2>
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
        </div>
      </section>
    </div>
  )
}
