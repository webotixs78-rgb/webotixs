'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, ExternalLink } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockPortfolio } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'

export default function PortfolioSection() {
  const [projects, setProjects] = useState<any[]>(mockPortfolio.slice(0, 3))

  useEffect(() => {
    getCMSData('portfolio').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setProjects(data.slice(0, 3))
      }
    })
  }, [])

  return (
    <section className="section-padding bg-background relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
            <span className="w-1.5 h-1.5 bg-primary-to rounded-full" />
            <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Our Work</span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-bold text-text-white mb-5">
            Projects That{' '}
            <span className="gradient-text">Speak Volumes</span>
          </h2>
          <p className="text-text-gray text-lg max-w-2xl mx-auto">
            A curated selection of our finest work — where strategy meets design and technology.
          </p>
        </ScrollReveal>

        {/* Projects */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {(projects || []).map((project: any, i: number) => (
            <ScrollReveal key={project.id || i} delay={i * 0.15}>
              <Link href={`/portfolio/${project.id || ''}`} className="group block">
                <div className="bg-background-card rounded-3xl overflow-hidden border border-border card-hover">
                  {/* Image placeholder */}
                  <div className="relative h-52 bg-gradient-to-br from-background-section to-background-secondary overflow-hidden">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-from/30 to-primary-to/30 flex items-center justify-center mx-auto mb-3">
                          <span className="text-2xl font-bold gradient-text">{(project.title || 'P')[0]}</span>
                        </div>
                        <div className="text-text-gray/50 text-xs">{project.industry}</div>
                      </div>
                    </div>
                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    {/* External link badge */}
                    {project.live_url && (
                      <div className="absolute top-4 right-4 w-8 h-8 glass rounded-lg flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <ExternalLink size={14} className="text-text-white" />
                      </div>
                    )}
                    {/* Industry badge */}
                    <div className="absolute bottom-4 left-4">
                      <span className="px-3 py-1 glass rounded-full text-xs text-text-gray border border-border/60">
                        {project.industry}
                      </span>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    <h3 className="font-display text-lg font-bold text-text-white mb-2 group-hover:gradient-text transition-all duration-300">
                      {project.title}
                    </h3>
                    <p className="text-text-gray text-sm mb-4 line-clamp-2">{project.description}</p>

                    {/* Results */}
                    <div className="flex flex-wrap gap-2 mb-4">
                      {(project.results || []).slice(0, 2).map((result: string, j: number) => (
                        <span key={j} className="px-2.5 py-1 bg-success/10 border border-success/20 rounded-lg text-success text-xs font-medium">
                          {result}
                        </span>
                      ))}
                    </div>

                    {/* Tech stack */}
                    <div className="flex flex-wrap gap-1.5">
                      {(project.technologies || []).slice(0, 4).map((tech: string, idx: number) => (
                        <span key={idx} className="px-2 py-0.5 bg-background rounded-lg text-text-gray text-xs border border-border/50">
                          {tech}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </Link>
            </ScrollReveal>
          ))}
        </div>

        {/* CTA */}
        <ScrollReveal className="text-center mt-12">
          <Link
            href="/portfolio"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl hover:shadow-glow-sm transition-all duration-300 hover:scale-105"
          >
            View All Projects <ArrowRight size={16} />
          </Link>
        </ScrollReveal>
      </div>
    </section>
  )
}
