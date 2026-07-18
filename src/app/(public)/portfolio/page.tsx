import Link from 'next/link'
import { ExternalLink, ArrowRight } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Our Portfolio',
  description: 'Explore the portfolio of Webotixs. Discover our successful fintech, real estate, design and development projects.',
}

export default async function PortfolioPage() {
  const projects: any[] = await getCMSData('portfolio')

  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-primary-to rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Our Works</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Elite Digital <span className="gradient-text">Showcase</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              Explore the results-driven applications, beautiful marketing websites, and robust architectures we have built.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Projects Grid */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {projects.map((project, i) => (
              <ScrollReveal key={project.id} delay={i * 0.1}>
                <Link href={`/portfolio/${project.id}`} className="group block h-full">
                  <div className="h-full bg-background-card rounded-3xl overflow-hidden border border-border card-hover flex flex-col justify-between">
                    <div>
                      {/* Image placeholder or real image */}
                      <div className="relative h-56 bg-gradient-to-br from-background-section to-background-secondary overflow-hidden">
                        {project.thumbnail ? (
                          <img src={project.thumbnail} alt={project.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-center">
                              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-from/20 to-primary-to/20 flex items-center justify-center mx-auto mb-4 border border-primary/20">
                                <span className="text-3xl font-bold gradient-text">{project.title?.[0] || 'P'}</span>
                              </div>
                              <span className="text-text-gray text-xs">{project.client}</span>
                            </div>
                          </div>
                        )}
                        {/* Hover Overlay */}
                        <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        {/* Industry badge */}
                        <div className="absolute bottom-4 left-4 z-10">
                          <span className="px-3 py-1 glass rounded-full text-xs text-text-white border border-border/60">
                            {project.industry}
                          </span>
                        </div>
                        {/* Link badge */}
                        {project.live_url && (
                          <div className="absolute top-4 right-4 w-9 h-9 glass rounded-xl flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300">
                            <ExternalLink size={14} className="text-text-white" />
                          </div>
                        )}
                      </div>

                      {/* Content */}
                      <div className="p-7">
                        <h2 className="font-display text-xl font-bold text-text-white mb-2 group-hover:gradient-text transition-colors duration-300">
                          {project.title}
                        </h2>
                        <p className="text-text-gray text-sm mb-5 leading-relaxed line-clamp-3">
                          {project.description}
                        </p>

                        {/* Performance metrics / Results */}
                        <div className="flex flex-wrap gap-2 mb-5">
                          {(project.results || []).map((res: string, j: number) => (
                            <span key={j} className="px-2.5 py-1 bg-success/10 border border-success/20 rounded-lg text-success text-xs font-semibold">
                              {res}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Tech & Action */}
                    <div className="px-7 pb-7">
                      <div className="flex flex-wrap gap-1.5 mb-6">
                        {(project.technologies || []).map((tech: string, idx: number) => (
                          <span key={idx} className="px-2 py-0.5 bg-background rounded-lg text-text-gray text-xs border border-border/50">
                            {tech}
                          </span>
                        ))}
                      </div>

                      <div className="flex items-center gap-2 text-primary text-sm font-semibold group-hover:underline">
                        View Project Breakdown
                        <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </div>
                </Link>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
