import Link from 'next/link'
import { ExternalLink, ArrowRight } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'
import { slugify } from '@/lib/utils'
import PortfolioGridClient from '@/components/public/portfolio/PortfolioGridClient'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Our Work & Case Studies',
  description: 'Explore high-impact enterprise web applications, e-commerce platforms, and mobile products delivered by Webotixs.',
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
              <span className="w-1.5 h-1.5 bg-primary-from rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Proven Track Record</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Our Digital <span className="gradient-text">Masterpieces</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              We don’t just write code — we engineer revenue-generating assets. Explore our enterprise case studies and quantifiable client outcomes.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Portfolio Grid */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <PortfolioGridClient initialData={projects} />
        </div>
      </section>
    </div>
  )
}
