import { Github, Linkedin, Mail } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'

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
              Meet the <span className="gradient-text">Experts</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              We bring together talented engineers, creative artists, and business consultants to create premium digital products.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Team Matrix */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {team.map((member, i) => (
              <ScrollReveal key={member.id} delay={i * 0.1}>
                <div className="group bg-background-card rounded-3xl p-8 border border-border card-hover text-center flex flex-col justify-between h-full">
                  <div>
                    {/* Avatar or Photo */}
                    <div className="relative w-24 h-24 mx-auto mb-6">
                      {member.photo ? (
                        <img src={member.photo} alt={member.name} className="w-24 h-24 rounded-2xl object-cover border border-primary/20 group-hover:border-primary/45 group-hover:shadow-glow-sm transition-all duration-300" />
                      ) : (
                        <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center border border-primary/20 group-hover:border-primary/45 group-hover:shadow-glow-sm transition-all duration-300">
                          <span className="font-display font-bold text-3xl gradient-text">
                            {member.name.split(' ').map((n: string) => n[0]).join('')}
                          </span>
                        </div>
                      )}
                      <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-success rounded-full border-2 border-background-card" />
                    </div>

                    <h2 className="font-display text-xl font-bold text-text-white mb-1 group-hover:gradient-text transition-colors duration-300">
                      {member.name}
                    </h2>
                    <p className="text-primary text-sm font-semibold mb-4">{member.position}</p>
                    <p className="text-text-gray text-sm leading-relaxed mb-6">
                      {member.bio}
                    </p>
                  </div>

                  {/* Social links */}
                  <div className="flex items-center justify-center gap-3 border-t border-border/40 pt-5">
                    {member.linkedin && (
                      <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="w-9 h-9 glass rounded-xl flex items-center justify-center text-text-gray hover:text-primary transition-colors border border-border">
                        <Linkedin size={15} />
                      </a>
                    )}
                    {member.github && (
                      <a href={member.github} target="_blank" rel="noopener noreferrer" className="w-9 h-9 glass rounded-xl flex items-center justify-center text-text-gray hover:text-primary transition-colors border border-border">
                        <Github size={15} />
                      </a>
                    )}
                    {member.email && (
                      <a href={`mailto:${member.email}`} className="w-9 h-9 glass rounded-xl flex items-center justify-center text-text-gray hover:text-primary transition-colors border border-border">
                        <Mail size={15} />
                      </a>
                    )}
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
