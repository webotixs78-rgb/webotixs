import Link from 'next/link'
import { ArrowRight, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, Check } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'
import { slugify } from '@/lib/utils'
import ServicesGridClient from '@/components/public/services/ServicesGridClient'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Enterprise Digital Services & Engineering',
  description: 'Custom software development, UI/UX design, cloud architecture, and digital transformation for global brands.',
}

export default async function ServicesPage() {
  const services: any[] = await getCMSData('services')

  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-primary-from rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Enterprise Capabilities</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Our Digital <span className="gradient-text">Expertise</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              We design, build, and optimize elite digital products that convert. Tailored from the ground up to fit your exact specifications and drive revenue.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Services Grid */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ServicesGridClient initialData={services} />
        </div>
      </section>
    </div>
  )
}
