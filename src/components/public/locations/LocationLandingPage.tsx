'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  ArrowRight,
  Check,
  MapPin,
  Palette,
  Globe,
  TrendingUp,
  Cloud,
  ShieldCheck,
  Bot,
  Database,
  Zap,
  Code,
  ShoppingCart,
  Layout,
  Sparkles,
  ChevronDown,
  Building2,
  Server,
  Cpu,
  Layers,
  CheckCircle2,
} from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'
import TechLogoSlider from '@/components/public/home/TechLogoSlider'
import PortfolioSection from '@/components/public/home/PortfolioSection'
import WhyChooseUsSection from '@/components/public/home/WhyChooseUsSection'
import { LocationConfig } from '@/lib/data/locationData'

const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Palette,
  Globe,
  TrendingUp,
  MapPin,
  Cloud,
  ShieldCheck,
  Bot,
  Database,
  Zap,
  Code,
  ShoppingCart,
  Layout,
}

interface Props {
  config: LocationConfig
}

export default function LocationLandingPage({ config }: Props) {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0)

  const toggleFaq = (index: number) => {
    setOpenFaqIndex(openFaqIndex === index ? null : index)
  }

  // Generate JSON-LD Schema
  const schemaOrg = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'ProfessionalService',
        '@id': `${config.canonicalUrl}#service`,
        name: `Webotixs - ${config.h1}`,
        url: config.canonicalUrl,
        image: 'https://www.webotixs.com/og-image.jpg',
        description: config.metaDescription,
        priceRange: '$$$',
        areaServed: {
          '@type': 'City',
          name: config.city,
          containedIn: config.state,
        },
        provider: {
          '@type': 'Organization',
          name: 'Webotixs',
          url: 'https://www.webotixs.com',
          logo: 'https://www.webotixs.com/icon.png',
        },
      },
      {
        '@type': 'BreadcrumbList',
        '@id': `${config.canonicalUrl}#breadcrumb`,
        itemListElement: [
          {
            '@type': 'ListItem',
            position: 1,
            name: 'Home',
            item: 'https://www.webotixs.com',
          },
          {
            '@type': 'ListItem',
            position: 2,
            name: 'Locations',
            item: 'https://www.webotixs.com/locations',
          },
          {
            '@type': 'ListItem',
            position: 3,
            name: `${config.city}, ${config.stateAbbr}`,
            item: config.canonicalUrl,
          },
        ],
      },
      {
        '@type': 'FAQPage',
        '@id': `${config.canonicalUrl}#faq`,
        mainEntity: config.faqs.map((faq) => ({
          '@type': 'Question',
          name: faq.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: faq.answer,
          },
        })),
      },
    ],
  }

  return (
    <div className="bg-background min-h-screen relative overflow-hidden pt-20">
      {/* Schema Injection */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaOrg) }}
      />

      {/* 1. HERO SECTION */}
      <section className="relative py-20 md:py-28 overflow-hidden border-b border-border/40">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <ScrollReveal className="text-center max-w-4xl mx-auto space-y-6">
            {/* Eyebrow Pill */}
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-glow-sm" />
              <span className="text-text-gray text-xs font-mono font-bold uppercase tracking-widest">
                {config.eyebrow}
              </span>
            </div>

            {/* H1 */}
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl font-bold text-text-white tracking-tight leading-[1.15]">
              {config.h1.split(config.city)[0]}
              <span className="gradient-text">{config.city}</span>
              {config.h1.split(config.city)[1]}
            </h1>

            {/* Supporting Copy */}
            <p className="text-text-gray text-lg sm:text-xl leading-relaxed max-w-3xl mx-auto font-normal">
              {config.heroSupport}
            </p>

            {/* Location Reference Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-xl border border-cyan-500/30 text-xs font-semibold text-cyan-300">
              <MapPin size={14} className="text-cyan-400 shrink-0" />
              <span>{config.locationReferenceBadge}</span>
            </div>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
              <Link
                href="/contact"
                aria-label={`Start Your Web Design Project in ${config.city}, ${config.stateAbbr}`}
                className="btn-float-rtl px-8 py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-bold rounded-2xl shadow-glow-md hover:shadow-glow-lg transition-all duration-300 hover:scale-105 flex items-center gap-2"
              >
                Start Your Project <ArrowRight size={18} />
              </Link>
              <a
                href="#services"
                aria-label={`Explore Our Digital Services in ${config.city}`}
                className="btn-float-rtl-glass px-8 py-4 glass border border-border text-text-white font-bold rounded-2xl hover:border-primary/50 transition-all duration-300"
              >
                Explore Our Services
              </a>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* 2. TRUST / STATS SECTION */}
      <section className="py-14 bg-background-secondary border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { stat: '200+', label: 'Digital Projects Delivered', icon: Globe },
              { stat: '98%', label: 'Client Satisfaction Rate', icon: CheckCircle2 },
              { stat: '8+', label: 'Years Agency Experience', icon: Building2 },
              { stat: '50+', label: 'Engineers & Designers', icon: Cpu },
            ].map((item, idx) => (
              <ScrollReveal key={idx} delay={idx * 0.1}>
                <GlowingGlassCard className="p-6 text-center rounded-2xl glass border border-border/50">
                  <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-3 text-primary">
                    <item.icon size={20} />
                  </div>
                  <div className="font-display text-3xl sm:text-4xl font-bold gradient-text mb-1">
                    {item.stat}
                  </div>
                  <div className="text-text-gray text-xs font-semibold uppercase tracking-wider">
                    {item.label}
                  </div>
                </GlowingGlassCard>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* 3. INTRODUCTION SECTION */}
      <section className="py-20 bg-background relative border-b border-border/40">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="space-y-6">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
              <Sparkles size={14} className="text-cyan-400" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">
                Regional Growth Partner
              </span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white">
              {config.introHeading}
            </h2>
            <div className="space-y-5 text-text-gray text-base sm:text-lg leading-relaxed pt-2">
              {config.introParagraphs.map((para, i) => (
                <p key={i}>{para}</p>
              ))}
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* 4. SERVICES SECTION */}
      <section id="services" className="py-24 bg-background-secondary border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-4">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Capabilities</span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-4">
              {config.servicesHeading}
            </h2>
            <p className="text-text-gray text-lg max-w-2xl mx-auto">
              {config.servicesSubheading}
            </p>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {config.services.map((srv, i) => {
              const Icon = iconMap[srv.icon] ?? Globe
              return (
                <ScrollReveal key={i} delay={i * 0.05}>
                  <Link
                    href={srv.href}
                    aria-label={`Learn more about ${srv.title} in ${config.city}`}
                    className="group block h-full"
                  >
                    <GlowingGlassCard className="h-full bg-background-card/90 rounded-3xl p-7 border border-border/60 flex flex-col justify-between hover:border-primary/50 transition-all duration-300">
                      <div>
                        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-from/20 to-primary-to/20 flex items-center justify-center border border-primary/20 mb-5 group-hover:border-primary/50 transition-all">
                          <Icon size={22} className="text-primary" />
                        </div>
                        <div className="font-display text-xl font-bold text-text-white mb-2.5 group-hover:gradient-text transition-colors">
                          {srv.title}
                        </div>
                        <p className="text-text-gray text-sm leading-relaxed mb-6">
                          {srv.description}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 text-primary text-xs font-semibold group-hover:underline">
                        <span>Learn More</span>
                        <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                      </div>
                    </GlowingGlassCard>
                  </Link>
                </ScrollReveal>
              )
            })}
          </div>
        </div>
      </section>

      {/* 5. WEB DESIGN & DEVELOPMENT SECTION */}
      <section className="py-24 bg-background border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <ScrollReveal className="lg:col-span-7 space-y-6">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
                <Code size={14} className="text-cyan-400" />
                <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Engineering Excellence</span>
              </div>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
                {config.webDevHeading}
              </h2>
              <div className="space-y-4 text-text-gray text-base leading-relaxed">
                {config.webDevCopy.map((para, idx) => (
                  <p key={idx}>{para}</p>
                ))}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
                {config.webDevFeatures.map((feat, idx) => (
                  <div key={idx} className="flex items-center gap-2.5 p-3 rounded-xl glass border border-border/40 text-text-white text-xs font-semibold">
                    <span className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center text-primary shrink-0">
                      <Check size={12} />
                    </span>
                    <span>{feat}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>

            <ScrollReveal className="lg:col-span-5">
              <GlowingGlassCard className="p-8 rounded-3xl border border-border/60 space-y-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-cyan-500/20 border border-blue-500/30 flex items-center justify-center text-cyan-400">
                  <Layers size={28} />
                </div>
                <div className="font-display text-2xl font-bold text-white">
                  Modern Tech Architecture
                </div>
                <p className="text-text-gray text-sm leading-relaxed">
                  We build custom web solutions with Next.js, React, Supabase, and WordPress designed for high availability, fast indexing, and seamless scalability.
                </p>
                <Link
                  href="/services/web-design-development"
                  aria-label="Explore Web Design & Development Services"
                  className="inline-flex items-center gap-2 text-primary font-bold text-sm hover:underline"
                >
                  Explore Web Engineering <ArrowRight size={16} />
                </Link>
              </GlowingGlassCard>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* 6. SEO SECTION */}
      <section className="py-24 bg-background-secondary border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <ScrollReveal className="lg:col-span-5 order-2 lg:order-1">
              <GlowingGlassCard className="p-8 rounded-3xl border border-border/60 space-y-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-primary/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                  <TrendingUp size={28} />
                </div>
                <div className="font-display text-2xl font-bold text-white">
                  Organic Search Dominance
                </div>
                <p className="text-text-gray text-sm leading-relaxed">
                  Our search strategies focus on long-term organic authority, technical compliance, local search maps, and measurable lead revenue growth.
                </p>
                <Link
                  href="/services/seo-digital-marketing"
                  aria-label="Explore SEO & Digital Marketing Services"
                  className="inline-flex items-center gap-2 text-primary font-bold text-sm hover:underline"
                >
                  Explore SEO Services <ArrowRight size={16} />
                </Link>
              </GlowingGlassCard>
            </ScrollReveal>

            <ScrollReveal className="lg:col-span-7 order-1 lg:order-2 space-y-6">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
                <TrendingUp size={14} className="text-cyan-400" />
                <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Search Engine Visibility</span>
              </div>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
                {config.seoHeading}
              </h2>
              <div className="space-y-4 text-text-gray text-base leading-relaxed">
                {config.seoCopy.map((para, idx) => (
                  <p key={idx}>{para}</p>
                ))}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
                {config.seoCapabilities.map((cap, idx) => (
                  <div key={idx} className="flex items-center gap-2.5 p-3 rounded-xl glass border border-border/40 text-text-white text-xs font-semibold">
                    <span className="w-5 h-5 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 shrink-0">
                      <Check size={12} />
                    </span>
                    <span>{cap}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* 7. HOSTING & MAINTENANCE SECTION */}
      <section className="py-24 bg-background border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <ScrollReveal className="lg:col-span-7 space-y-6">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
                <Cloud size={14} className="text-cyan-400" />
                <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Managed Infrastructure</span>
              </div>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
                {config.hostingHeading}
              </h2>
              <div className="space-y-4 text-text-gray text-base leading-relaxed">
                {config.hostingCopy.map((para, idx) => (
                  <p key={idx}>{para}</p>
                ))}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
                {config.hostingFeatures.map((feat, idx) => (
                  <div key={idx} className="flex items-center gap-2.5 p-3 rounded-xl glass border border-border/40 text-text-white text-xs font-semibold">
                    <span className="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 shrink-0">
                      <Check size={12} />
                    </span>
                    <span>{feat}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>

            <ScrollReveal className="lg:col-span-5">
              <GlowingGlassCard className="p-8 rounded-3xl border border-border/60 space-y-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-cyan-500/20 border border-blue-500/30 flex items-center justify-center text-cyan-400">
                  <Server size={28} />
                </div>
                <div className="font-display text-2xl font-bold text-white">
                  Cloud Infrastructure & Support
                </div>
                <p className="text-text-gray text-sm leading-relaxed">
                  Keep your digital platforms running at top speeds with managed cloud hosting, daily automated backups, and routine security maintenance.
                </p>
                <Link
                  href="/services/cloud-devops"
                  aria-label="Explore Cloud & Website Maintenance Services"
                  className="inline-flex items-center gap-2 text-primary font-bold text-sm hover:underline"
                >
                  Explore Hosting & Maintenance <ArrowRight size={16} />
                </Link>
              </GlowingGlassCard>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* 8. AI CHATBOT & AUTOMATION SECTION */}
      <section className="py-24 bg-background-secondary border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <ScrollReveal className="lg:col-span-5 order-2 lg:order-1">
              <GlowingGlassCard className="p-8 rounded-3xl border border-border/60 space-y-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/20 to-primary-to/20 border border-primary/30 flex items-center justify-center text-primary">
                  <Bot size={28} />
                </div>
                <div className="font-display text-2xl font-bold text-white">
                  24/7 AI Lead Automation
                </div>
                <p className="text-text-gray text-sm leading-relaxed">
                  Never miss a prospective customer inquiry. Our custom AI chatbots engage visitors, answer questions, and schedule consultations automatically.
                </p>
                <Link
                  href="/services/web-design-development"
                  aria-label="Explore AI Chatbot Solutions"
                  className="inline-flex items-center gap-2 text-primary font-bold text-sm hover:underline"
                >
                  Explore AI Automation <ArrowRight size={16} />
                </Link>
              </GlowingGlassCard>
            </ScrollReveal>

            <ScrollReveal className="lg:col-span-7 order-1 lg:order-2 space-y-6">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
                <Bot size={14} className="text-cyan-400" />
                <span className="text-text-gray text-xs font-medium uppercase tracking-wider">AI Innovation</span>
              </div>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
                {config.aiHeading}
              </h2>
              <div className="space-y-4 text-text-gray text-base leading-relaxed">
                {config.aiCopy.map((para, idx) => (
                  <p key={idx}>{para}</p>
                ))}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
                {config.aiUseCases.map((uc, idx) => (
                  <div key={idx} className="flex items-center gap-2.5 p-3 rounded-xl glass border border-border/40 text-text-white text-xs font-semibold">
                    <span className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center text-primary shrink-0">
                      <Check size={12} />
                    </span>
                    <span>{uc}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* 9. CRM SERVICES SECTION */}
      <section className="py-24 bg-background border-b border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <ScrollReveal className="lg:col-span-7 space-y-6">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
                <Database size={14} className="text-cyan-400" />
                <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Business Intelligence</span>
              </div>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
                {config.crmHeading}
              </h2>
              <div className="space-y-4 text-text-gray text-base leading-relaxed">
                {config.crmCopy.map((para, idx) => (
                  <p key={idx}>{para}</p>
                ))}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
                {config.crmFeatures.map((feat, idx) => (
                  <div key={idx} className="flex items-center gap-2.5 p-3 rounded-xl glass border border-border/40 text-text-white text-xs font-semibold">
                    <span className="w-5 h-5 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 shrink-0">
                      <Check size={12} />
                    </span>
                    <span>{feat}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>

            <ScrollReveal className="lg:col-span-5">
              <GlowingGlassCard className="p-8 rounded-3xl border border-border/60 space-y-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-primary/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                  <Database size={28} />
                </div>
                <div className="font-display text-2xl font-bold text-white">
                  Custom Pipeline Automation
                </div>
                <p className="text-text-gray text-sm leading-relaxed">
                  Organize leads, track sales pipelines, and automate task follow-ups with tailored CRM development and third-party integrations.
                </p>
                <Link
                  href="/services/web-design-development"
                  aria-label="Explore CRM & Software Development Services"
                  className="inline-flex items-center gap-2 text-primary font-bold text-sm hover:underline"
                >
                  Explore CRM Solutions <ArrowRight size={16} />
                </Link>
              </GlowingGlassCard>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* 10. WHY WEBOTIXS SECTION */}
      <WhyChooseUsSection />

      {/* 11. TECHNOLOGY ECOSYSTEM SECTION */}
      <TechLogoSlider />

      {/* 12. PORTFOLIO / CASE STUDIES SECTION */}
      <PortfolioSection />

      {/* 13. LOCAL RELEVANCE SECTION */}
      <section className="py-20 bg-background-secondary border-b border-border/40 relative">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center space-y-6">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
              <MapPin size={14} className="text-cyan-400" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Regional Coverage</span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
              {config.localAreaHeading}
            </h2>
            <p className="text-text-gray text-base sm:text-lg max-w-3xl mx-auto">
              {config.localAreaIntro}
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3 pt-4">
              {config.localAreas.map((area, idx) => (
                <div
                  key={idx}
                  className="px-4 py-2.5 glass border border-border/60 rounded-2xl text-text-white text-sm font-semibold flex items-center gap-2"
                >
                  <MapPin size={14} className="text-cyan-400" />
                  <span>{area}</span>
                </div>
              ))}
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* 14. FAQ SECTION */}
      <section className="py-24 bg-background border-b border-border/40 relative">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center mb-16 space-y-3">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 glass rounded-full border border-border/60">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Got Questions?</span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-text-white">
              {config.faqsHeading}
            </h2>
          </ScrollReveal>

          <div className="space-y-4">
            {config.faqs.map((faq, idx) => {
              const isOpen = openFaqIndex === idx
              return (
                <ScrollReveal key={idx} delay={idx * 0.05}>
                  <div className="glass border border-border/60 rounded-2xl overflow-hidden transition-all duration-300">
                    <button
                      onClick={() => toggleFaq(idx)}
                      className="w-full p-6 text-left flex items-center justify-between gap-4 font-display text-lg font-bold text-text-white hover:text-primary transition-colors"
                      aria-expanded={isOpen}
                    >
                      <span>{faq.question}</span>
                      <ChevronDown
                        size={20}
                        className={`text-text-gray transition-transform duration-300 shrink-0 ${
                          isOpen ? 'rotate-180 text-primary' : ''
                        }`}
                      />
                    </button>
                    {isOpen && (
                      <div className="px-6 pb-6 text-text-gray text-sm sm:text-base leading-relaxed border-t border-border/30 pt-4">
                        {faq.answer}
                      </div>
                    )}
                  </div>
                </ScrollReveal>
              )
            })}
          </div>
        </div>
      </section>

      {/* 15. FINAL CTA SECTION */}
      <section className="py-24 bg-background-secondary relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal>
            <div className="glass rounded-3xl p-8 md:p-14 text-center relative overflow-hidden border border-border/60">
              <div className="absolute inset-0 bg-gradient-to-br from-primary-from/15 to-primary-to/15 rounded-3xl pointer-events-none" />
              <div className="relative z-10 space-y-6 max-w-3xl mx-auto">
                <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white">
                  {config.ctaHeading.split(config.city)[0]}
                  <span className="gradient-text">{config.city}?</span>
                </h2>
                <p className="text-text-gray text-lg leading-relaxed">
                  {config.ctaSupport}
                </p>
                <div className="pt-4">
                  <Link
                    href="/contact"
                    aria-label={`Start Your Project with Webotixs in ${config.city}`}
                    className="btn-float-rtl inline-flex items-center gap-2 px-9 py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-bold rounded-2xl shadow-glow-md hover:shadow-glow-lg transition-all duration-300 hover:scale-105"
                  >
                    Start Your Project <ArrowRight size={18} />
                  </Link>
                </div>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </div>
  )
}
