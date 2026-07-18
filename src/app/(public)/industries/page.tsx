import Link from 'next/link'
import { Landmark, ShoppingBag, ShieldCheck, GraduationCap, Truck, HeartPulse } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'

export const metadata = {
  title: 'Industries We Serve',
  description: 'Explore the industries Webotixs builds software for, including Fintech, E-Commerce, Healthcare, Education, Logistics and Real Estate.',
}

const industries = [
  {
    icon: Landmark,
    title: 'Fintech & Banking',
    description: 'Secure, high-frequency transaction applications, custom mobile banking, and automated investment portfolios.',
    benefits: ['RSC Secured Encryption', '99.99% Transaction Uptime', 'Compliance & Regulations Built-in'],
  },
  {
    icon: ShoppingBag,
    title: 'E-Commerce & Retail',
    description: 'High-speed headless commerce stores, custom dashboards, inventory managers, and multi-currency checkouts.',
    benefits: ['Sub-second Page Load Times', 'AI Recommender Feeds', 'One-Click Apple & Stripe Payments'],
  },
  {
    icon: HeartPulse,
    title: 'Healthcare & Telemedicine',
    description: 'HIPAA-compliant patient portals, doctor appointment scheduling, secure records databases, and high-definition video consults.',
    benefits: ['Fully HIPAA Compliant', 'Secure Encrypted Databases', 'Interactive Consultation Tools'],
  },
  {
    icon: GraduationCap,
    title: 'EdTech & E-Learning',
    description: 'Custom learning management systems (LMS), real-time virtual classrooms, progress monitoring dashboards, and payment portals.',
    benefits: ['Global Content Delivery Net (CDN)', 'Adaptive Student Pipelines', 'Automated Quizzing Systems'],
  },
  {
    icon: Truck,
    title: 'Logistics & Supply Chain',
    description: 'Real-time GPS mapping dashboards, fleet optimization calculations, shipment tracking apps, and warehouse inventory databases.',
    benefits: ['WebSocket Live Feeds', 'Optimized Routing APIs', 'Automated Logistics Reports'],
  },
]

export default function IndustriesPage() {
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {industries.map((ind, i) => {
              const Icon = ind.icon
              return (
                <ScrollReveal key={ind.title} delay={i * 0.1}>
                  <div className="bg-background-card border border-border/60 rounded-3xl p-8 hover:border-primary/40 transition-all duration-300 h-full flex flex-col justify-between">
                    <div>
                      <div className="w-14 h-14 bg-primary/10 rounded-2xl flex items-center justify-center mb-6 text-primary border border-primary/20">
                        <Icon size={24} />
                      </div>
                      <h2 className="font-display text-2xl font-bold text-text-white mb-3">{ind.title}</h2>
                      <p className="text-text-gray text-sm leading-relaxed mb-6">{ind.description}</p>

                      <div className="space-y-2 mb-8">
                        <h3 className="text-xs uppercase text-text-white font-bold tracking-wider mb-3">Key Solutions:</h3>
                        {ind.benefits.map((benefit, j) => (
                          <div key={j} className="flex items-center gap-2.5 text-text-gray text-xs">
                            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                            {benefit}
                          </div>
                        ))}
                      </div>
                    </div>

                    <Link
                      href="/contact"
                      className="inline-flex items-center justify-center gap-2 w-full py-3.5 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-102"
                    >
                      Discuss Your Industry Needs
                    </Link>
                  </div>
                </ScrollReveal>
              )
            })}
          </div>
        </div>
      </section>
    </div>
  )
}
