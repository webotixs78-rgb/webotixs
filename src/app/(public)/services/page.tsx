import Link from 'next/link'
import { ArrowRight, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, Check } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Our Services',
  description: 'Explore the digital services Webotixs offers, including custom web design, mobile app development, brand identity, SEO, cloud, and e-commerce setups.',
}

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Globe,
  Smartphone,
  Palette,
  ShoppingCart,
  TrendingUp,
  Cloud,
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
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">What We Offer</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Next-Gen Services <br /> for <span className="gradient-text">Modern Brands</span>
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
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {services.map((service, i) => {
              const Icon = iconMap[service.icon] ?? Globe
              return (
                <ScrollReveal key={service.id} delay={i * 0.1}>
                  <div className="h-full bg-background-card border border-border/60 rounded-3xl p-8 card-hover flex flex-col justify-between overflow-hidden relative">
                    <div>
                      {service.cover_image && (
                        <div className="h-40 -mx-8 -mt-8 mb-6 overflow-hidden border-b border-border/50">
                          <img src={service.cover_image} alt={service.title} className="w-full h-full object-cover" />
                        </div>
                      )}
                      {/* Icon */}
                      <div className="w-14 h-14 bg-gradient-to-br from-primary-from/20 to-primary-to/20 rounded-2xl flex items-center justify-center mb-6 border border-primary/20">
                        <Icon size={24} className="text-primary" />
                      </div>

                      <h2 className="font-display text-2xl font-bold text-text-white mb-3 hover:gradient-text transition-colors">
                        {service.title}
                      </h2>
                      <p className="text-text-gray text-sm leading-relaxed mb-6">
                        {service.long_description}
                      </p>

                      {/* Features */}
                      <ul className="space-y-2.5 mb-8">
                        {(service.features || []).map((feat: string, idx: number) => (
                          <li key={idx} className="flex items-center gap-2.5 text-text-gray text-sm">
                            <span className="w-5 h-5 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                              <Check size={12} />
                            </span>
                            {feat}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <Link
                      href={`/services/${service.slug}`}
                      className="inline-flex items-center justify-center gap-2 w-full py-3.5 glass border border-border hover:border-primary/50 text-text-white hover:text-primary font-semibold rounded-2xl transition-all duration-300 group"
                    >
                      Explore Service Detail
                      <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
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
