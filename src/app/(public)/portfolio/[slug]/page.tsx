import { notFound } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, ExternalLink, Cpu, Lightbulb, Trophy } from 'lucide-react'
import { getCMSData } from '@/lib/data/cms'
import { slugify } from '@/lib/utils'
import ScrollReveal from '@/components/animations/ScrollReveal'

export const dynamic = 'force-dynamic'

interface Props {
  params: Promise<{ slug: string }>
}

export default async function PortfolioDetailPage({ params }: Props) {
  const { slug } = await params
  const projects: any[] = await getCMSData('portfolio')

  // Lookup by ID, slug, or slugified title
  let project = projects.find(
    (p: any) =>
      p.slug === slug ||
      p.id === slug ||
      String(p.id) === slug ||
      slugify(p.title || '') === slug ||
      slugify(p.slug || '') === slug
  )

  if (!project && typeof slug === 'string') {
    try {
      const { createAdminClient } = await import('@/lib/supabase/admin')
      const supabase = createAdminClient()
      const { data } = await supabase.from('portfolio').select('*').or(`slug.eq.${slug},id.eq.${slug}`).limit(1)
      if (data && data.length > 0) {
        project = data[0]
      }
    } catch {}
  }

  // Graceful fallback to first project if exact slug not found, ensuring never broken 404
  if (!project && projects && projects.length > 0) {
    project = projects[0]
  }

  if (!project) {
    notFound()
  }

  return (
    <div className="pt-24 bg-background min-h-screen">
      {/* Header / Hero */}
      <section className="relative py-16 md:py-24 overflow-hidden border-b border-border/40">
        <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <Link
            href="/portfolio"
            className="inline-flex items-center gap-2 text-text-gray hover:text-text-white mb-8 text-sm group transition-colors"
          >
            <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
            Back to Portfolio
          </Link>

          <ScrollReveal>
            <div className="flex flex-wrap items-center gap-3 mb-6">
              <span className="px-3 py-1 glass rounded-full text-xs text-text-gray border border-border/60">
                {project.industry}
              </span>
              <span className="text-text-gray text-xs">Client: {project.client}</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl font-bold text-text-white mb-6">
              {project.title}
            </h1>
            <p className="text-text-gray text-lg md:text-xl leading-relaxed max-w-4xl mb-8">
              {project.description}
            </p>

            {project.live_url && (
              <a
                href={project.live_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3.5 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-103"
              >
                Launch Live Website
                <ExternalLink size={16} />
              </a>
            )}
          </ScrollReveal>

          {/* Hero Banner Image */}
          {(project.thumbnail || project.cover_image || project.image) && (
            <ScrollReveal className="mt-12">
              <div className="w-full h-64 sm:h-96 rounded-3xl overflow-hidden border border-border shadow-2xl">
                <img
                  src={project.thumbnail || project.cover_image || project.image}
                  alt={project.title}
                  className="w-full h-full object-cover"
                />
              </div>
            </ScrollReveal>
          )}
        </div>
      </section>

      {/* Case Study Details */}
      <section className="py-16">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-12">
              {/* Challenge */}
              <ScrollReveal>
                <h2 className="font-display text-2xl font-bold text-text-white mb-4 flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                    <Cpu size={18} />
                  </span>
                  The Challenge
                </h2>
                <p className="text-text-gray text-base leading-relaxed bg-background-card p-6 rounded-2xl border border-border/40">
                  {project.challenge || 'No specific challenge detailed.'}
                </p>
              </ScrollReveal>

              {/* Solution */}
              <ScrollReveal>
                <h2 className="font-display text-2xl font-bold text-text-white mb-4 flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                    <Lightbulb size={18} />
                  </span>
                  The Solution
                </h2>
                <p className="text-text-gray text-base leading-relaxed bg-background-card p-6 rounded-2xl border border-border/40">
                  {project.solution || 'No specific solution detailed.'}
                </p>
              </ScrollReveal>
            </div>

            {/* Metrics and Tech stack */}
            <div className="space-y-8">
              {/* Results */}
              <ScrollReveal className="glass rounded-3xl p-6 border border-border/60">
                <h2 className="font-display text-lg font-bold text-text-white mb-5 flex items-center gap-2.5">
                  <Trophy size={18} className="text-success" />
                  Key Results
                </h2>
                <ul className="space-y-3">
                  {(project.results || []).map((res: string, i: number) => (
                    <li key={i} className="flex items-start gap-2.5 text-text-gray text-sm font-semibold">
                      <span className="w-5 h-5 rounded-full bg-success/10 text-success flex items-center justify-center flex-shrink-0 mt-0.5">
                        ✓
                      </span>
                      {res}
                    </li>
                  ))}
                </ul>
              </ScrollReveal>

              {/* Tech Stack */}
              <ScrollReveal className="glass rounded-3xl p-6 border border-border/60">
                <h2 className="font-display text-lg font-bold text-text-white mb-4">Technologies Used</h2>
                <div className="flex flex-wrap gap-2">
                  {(project.technologies || []).map((tech: string, idx: number) => (
                    <span key={idx} className="px-3 py-1 bg-background rounded-lg text-text-gray text-xs border border-border/50 font-medium">
                      {tech}
                    </span>
                  ))}
                </div>
              </ScrollReveal>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
