import { notFound } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, Check, Compass, Workflow, ShieldCheck, Mail } from 'lucide-react'
import { getCMSData } from '@/lib/data/cms'
import ScrollReveal from '@/components/animations/ScrollReveal'

export const dynamic = 'force-dynamic'

interface Props {
  params: Promise<{ slug: string }>
}

export default async function ServiceDetailPage({ params }: Props) {
  const { slug } = await params
  const services: any[] = await getCMSData('services')
  const service = services.find((s: any) => s.slug === slug)

  if (!service) {
    notFound()
  }

  return (
    <div className="pt-24 bg-background min-h-screen">
      {/* Header Info */}
      <section className="relative py-16 md:py-24 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <Link
            href="/services"
            className="inline-flex items-center gap-2 text-text-gray hover:text-text-white mb-8 text-sm group transition-colors"
          >
            <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
            Back to All Services
          </Link>

          <ScrollReveal>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl font-bold text-text-white mb-6">
              {service.title}
            </h1>
            <p className="text-text-gray text-lg md:text-xl leading-relaxed max-w-4xl">
              {service.long_description}
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Main Details */}
      <section className="py-12 border-t border-border/40">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
            {/* Features & Key Offerings */}
            <div className="lg:col-span-2 space-y-12">
              <ScrollReveal>
                <h2 className="font-display text-2xl md:text-3xl font-bold text-text-white mb-6 flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                    <ShieldCheck size={18} />
                  </span>
                  Key Features & Capabilities
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {(service.features || []).map((feat: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-3 p-4 bg-background-card border border-border/50 rounded-2xl">
                      <span className="w-5 h-5 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0 mt-0.5">
                        <Check size={12} />
                      </span>
                      <span className="text-text-gray text-sm font-medium">{feat}</span>
                    </div>
                  ))}
                </div>
              </ScrollReveal>

              {/* Implementation Process */}
              <ScrollReveal>
                <h2 className="font-display text-2xl md:text-3xl font-bold text-text-white mb-6 flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                    <Workflow size={18} />
                  </span>
                  Our Process & Workflow
                </h2>
                <div className="relative border-l border-border/50 pl-6 space-y-8 ml-3">
                  {(service.process || []).map((step: any, idx: number) => (
                    <div key={idx} className="relative">
                      <span className="absolute -left-[35px] top-0.5 w-6.5 h-6.5 rounded-full bg-background-card border border-primary flex items-center justify-center text-xs font-bold text-primary">
                        {step.step || idx + 1}
                      </span>
                      <h3 className="font-display text-lg font-bold text-text-white mb-1.5">{step.title}</h3>
                      <p className="text-text-gray text-sm leading-relaxed">{step.description}</p>
                    </div>
                  ))}
                </div>
              </ScrollReveal>
            </div>

            {/* Sidebar Contact Card */}
            <div>
              <ScrollReveal className="glass rounded-3xl p-6 border border-border/60 sticky top-28 text-center">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-from/20 to-primary-to/20 flex items-center justify-center mx-auto mb-5 border border-primary/20">
                  <Compass size={22} className="text-primary" />
                </div>
                <h3 className="font-display text-xl font-bold text-text-white mb-2">Need a custom setup?</h3>
                <p className="text-text-gray text-sm mb-6 leading-relaxed">
                  Get in touch with our engineers for a custom quote or strategy session.
                </p>
                <Link
                  href="/contact"
                  className="flex items-center justify-center gap-2 w-full py-3.5 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-102 group"
                >
                  <Mail size={16} />
                  Contact Us
                </Link>
              </ScrollReveal>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
