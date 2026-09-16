import Link from 'next/link'
import { Landmark, ShoppingBag, ShieldCheck, GraduationCap, Truck, HeartPulse, Globe } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'
import IndustriesGridClient from '@/components/public/industries/IndustriesGridClient'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Industries We Serve',
  description: 'Explore the industries Webotixs builds software for, including Fintech, E-Commerce, Healthcare, Education, Logistics and Real Estate.',
}

export default async function IndustriesPage() {
  const industries: any[] = await getCMSData('industries')

  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-primary-from rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Expertise Areas</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Industries We <span className="gradient-text">Empower</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              We engineer industry-specific software solutions that solve actual business challenges, scale seamlessly, and guarantee performance.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Industries grid */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <IndustriesGridClient initialData={industries} />
        </div>
      </section>
    </div>
  )
}
