// v4 - borderless glassmorphism portfolio section
'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, ExternalLink, TrendingUp } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { mockPortfolio } from '@/lib/data/mock'
import { getCMSData } from '@/lib/data/cms'
import { slugify } from '@/lib/utils'

export default function PortfolioSection() {
  const [projects, setProjects] = useState<any[]>(mockPortfolio.slice(0, 3))

  useEffect(() => {
    let isMounted = true

    const loadData = () => {
      getCMSData('portfolio').then((data) => {
        if (!isMounted) return
        if (Array.isArray(data) && data.length > 0) {
          setProjects(data.slice(0, 3))
        }
      })
    }

    loadData()

    const handleStorage = () => {
      try {
        const local = localStorage.getItem('webotixs_cms_portfolio')
        if (local) {
          const parsed = JSON.parse(local)
          if (Array.isArray(parsed) && parsed.length > 0) {
            setProjects(parsed.slice(0, 3))
          }
        }
      } catch {}
    }

    handleStorage()
    window.addEventListener('storage', handleStorage)
    return () => {
      isMounted = false
      window.removeEventListener('storage', handleStorage)
    }
  }, [])

  return (
    <section style={{ padding: '110px 0', background: '#050816', position: 'relative', overflow: 'hidden' }}>
      {/* Ambient glow */}
      <div className="portfolio-ambient-glow" />

      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '0 24px', position: 'relative', zIndex: 10 }}>

        {/* Header */}
        <ScrollReveal className="portfolio-header-wrap">
          {/* Borderless Glass Badge Pill */}
          <div className="portfolio-badge-pill" style={{ marginLeft: 'auto', marginRight: 'auto', width: 'fit-content' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#22d3ee', display: 'inline-block', boxShadow: '0 0 12px #22d3ee', animation: 'pulse 2s infinite' }} />
            <span style={{ color: 'rgba(255,255,255,0.92)', fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em' }}>Our Work</span>
          </div>

          <h2 style={{ fontSize: 'clamp(2.2rem, 5vw, 3.6rem)', fontWeight: 800, color: '#ffffff', marginTop: '16px', marginBottom: '24px', letterSpacing: '-0.02em', lineHeight: 1.25 }}>
            Projects That{' '}
            <span style={{ background: 'linear-gradient(135deg, #60a5fa 0%, #22d3ee 50%, #a78bfa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
              Speak Volumes
            </span>
          </h2>
          <p style={{ color: '#94A3B8', fontSize: '1.08rem', maxWidth: '660px', margin: '0 auto', lineHeight: 1.75 }}>
            A curated selection of our finest work — where strategy meets design and cutting-edge technology.
          </p>
        </ScrollReveal>

        {/* Projects Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '36px' }}>
          {(projects || []).map((project: any, i: number) => {
            const projectHref = `/portfolio/${project.slug || slugify(project.title || '') || project.id || ''}`
            return (
              <ScrollReveal key={project.id || i} delay={i * 0.15} className="portfolio-card-wrap">
                <div className="portfolio-card">
                  {/* Image Container */}
                  <div style={{ position: 'relative', height: '250px', width: '100%', background: 'linear-gradient(135deg, #050816, #0A0E1F)', overflow: 'hidden' }}>
                    <Link
                      href={projectHref}
                      aria-label={`View ${project.title} Case Study`}
                      style={{ display: 'block', width: '100%', height: '100%' }}
                    >
                      {project.thumbnail || project.cover_image || project.image ? (
                        <img
                          src={project.thumbnail || project.cover_image || project.image}
                          alt={project.title}
                          className="portfolio-card-img"
                        />
                      ) : (
                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ width: '64px', height: '64px', borderRadius: '18px', background: 'rgba(59,130,246,0.15)', backdropFilter: 'blur(14px)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
                              <span style={{ fontSize: '1.5rem', fontWeight: 700, background: 'linear-gradient(90deg,#60a5fa,#22d3ee)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
                                {(project.title || 'P')[0]}
                              </span>
                            </div>
                            <div style={{ color: 'rgba(148,163,184,0.7)', fontSize: '11px', fontWeight: 500 }}>{project.industry}</div>
                          </div>
                        </div>
                      )}
                    </Link>

                    {/* Gradient Overlay */}
                    <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(10,14,31,0.95) 0%, rgba(0,0,0,0.2) 50%, transparent 100%)', opacity: 0.85, pointerEvents: 'none' }} />

                    {/* External Link Icon */}
                    {project.live_url && (
                      <a
                        href={project.live_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label={`Visit live website for ${project.title}`}
                        className="portfolio-ext-link"
                      >
                        <ExternalLink size={16} color="#ffffff" />
                      </a>
                    )}

                    {/* Category Glass Pill (Borderless) */}
                    <div style={{ position: 'absolute', bottom: '16px', left: '16px', zIndex: 10, pointerEvents: 'none' }}>
                      <span className="portfolio-category-pill">{project.industry || 'Web Application'}</span>
                    </div>
                  </div>

                  {/* Card Body */}
                  <div style={{ padding: '28px', display: 'flex', flexDirection: 'column', flex: 1, gap: '20px' }}>
                    <div>
                      <Link
                        href={projectHref}
                        style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', marginBottom: '10px', lineHeight: 1.35, display: 'block', textDecoration: 'none', transition: 'color 0.3s' }}
                        className="hover:text-cyan-400"
                      >
                        {project.title}
                      </Link>
                      <p style={{ color: '#94A3B8', fontSize: '0.875rem', lineHeight: 1.7, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                        {project.description}
                      </p>
                    </div>

                    {/* Statistics Notification Chips (Borderless Glass) */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {(project.results || []).slice(0, 2).map((result: string, j: number) => (
                        <div key={j} className="portfolio-result-chip">
                          <TrendingUp size={14} color="#22d3ee" style={{ flexShrink: 0 }} />
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{result}</span>
                        </div>
                      ))}
                    </div>

                    {/* Technology Tags (Borderless Glass Chips) */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {(project.technologies || []).slice(0, 4).map((tech: string, idx: number) => (
                        <span key={idx} className="portfolio-tech-tag">{tech}</span>
                      ))}
                    </div>

                    {/* Dedicated View Case Study Link */}
                    <div style={{ paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                      <Link
                        href={projectHref}
                        aria-label={`View case study: ${project.title}`}
                        style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', color: '#22d3ee', fontSize: '0.875rem', fontWeight: 600, textDecoration: 'none' }}
                        className="hover:underline"
                      >
                        <span>View Case Study</span>
                        <ArrowRight size={14} />
                      </Link>
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            )
          })}
        </div>

        {/* CTA */}
        <ScrollReveal className="portfolio-cta-wrap">
          <Link
            href="/portfolio"
            aria-label="Explore full Webotixs client portfolio"
            style={{ display: 'inline-flex', alignItems: 'center', gap: '10px', padding: '16px 36px', background: 'linear-gradient(135deg, #2563eb, #06b6d4)', color: '#ffffff', fontWeight: 700, fontSize: '0.875rem', borderRadius: '16px', textDecoration: 'none', boxShadow: '0 10px 30px rgba(37,99,235,0.4)', transition: 'transform 0.3s, box-shadow 0.3s', border: 'none' }}
          >
            View All Projects <ArrowRight size={16} />
          </Link>
        </ScrollReveal>
      </div>
    </section>
  )
}
