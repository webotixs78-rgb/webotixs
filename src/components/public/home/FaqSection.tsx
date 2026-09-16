// v4 - borderless glassmorphism FAQ section
'use client'

import { useState } from 'react'
import { ChevronDown, HelpCircle, Sparkles } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'

interface FAQItem {
  question: string
  answer: string
}

const faqList: FAQItem[] = [
  {
    question: 'What web development services does Webotixs provide?',
    answer:
      'Webotixs provides end-to-end digital solutions including custom web design & development, Next.js web applications, mobile app development (iOS & Android), headless e-commerce platforms (Shopify Plus & WooCommerce), brand identity design, API development, enterprise cloud architecture, and ongoing SEO & performance optimization.',
  },
  {
    question: 'How long does a custom web or mobile app development project take?',
    answer:
      'Project timelines depend on complexity. A bespoke corporate website or landing page typically takes 2 to 4 weeks. Full-scale custom web applications, SaaS portals, or native mobile apps generally take 6 to 12 weeks from initial strategy & discovery to final launch.',
  },
  {
    question: 'What technology stack do you use for high-performance applications?',
    answer:
      'We build with modern, production-tested technologies including Next.js, React, TypeScript, Tailwind CSS, Node.js, Python, Supabase, PostgreSQL, GraphQL, Docker, and AWS edge serverless infrastructure to guarantee sub-second load times and bank-grade security.',
  },
  {
    question: 'How do you ensure our website remains fast, secure, and up-to-date after launch?',
    answer:
      'We offer comprehensive 24/7 post-launch support and SLA maintenance plans. This includes continuous core dependency updates, security vulnerability patching, daily automated database backups, speed monitoring, and dedicated engineering assistance whenever you need updates.',
  },
  {
    question: 'Can you redesign our existing website without losing our search engine (SEO) rankings?',
    answer:
      'Yes, SEO preservation is a top priority. We perform a complete technical SEO audit prior to redesign, implement strict 301 URL redirect mapping, maintain existing high-ranking URL structures, and optimize structured JSON-LD schema to protect and boost your organic search rankings.',
  },
  {
    question: 'How does Webotixs structure project pricing and billing models?',
    answer:
      'We offer flexible billing models tailored to your requirements: Fixed-Price Milestone Contracts for clearly defined deliverables, and Monthly Dedicated Team Retainers for continuous product development. All proposals are transparent with no hidden fees.',
  },
]

export default function FaqSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(0)

  const toggle = (index: number) => {
    setOpenIndex((prev) => (prev === index ? null : index))
  }

  const faqSchema = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: faqList.map((faq) => ({
      '@type': 'Question',
      name: faq.question,
      acceptedAnswer: { '@type': 'Answer', text: faq.answer },
    })),
  }

  return (
    <section id="faq" style={{ padding: '110px 0', background: '#050816', position: 'relative', overflow: 'hidden' }}>
      {/* Schema JSON-LD */}
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(faqSchema) }} />

      {/* Ambient glow */}
      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: '750px', height: '420px', background: 'radial-gradient(circle, rgba(37,99,235,0.14) 0%, rgba(6,182,212,0.08) 50%, transparent 80%)', filter: 'blur(150px)', borderRadius: '9999px', pointerEvents: 'none' }} />

      <div style={{ maxWidth: '920px', margin: '0 auto', padding: '0 24px', position: 'relative', zIndex: 10 }}>

        {/* Header */}
        <ScrollReveal className="faq-header-wrap">
          {/* Borderless Glass Badge Pill */}
          <div className="faq-badge-pill" style={{ marginLeft: 'auto', marginRight: 'auto', width: 'fit-content' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#22d3ee', display: 'inline-block', boxShadow: '0 0 12px #22d3ee', animation: 'pulse 2s infinite' }} />
            <span style={{ color: 'rgba(255,255,255,0.92)', fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em' }}>Frequently Asked Questions</span>
          </div>

          <h2 style={{ fontSize: 'clamp(2.2rem, 5vw, 3.2rem)', fontWeight: 800, color: '#ffffff', marginTop: '16px', marginBottom: '24px', letterSpacing: '-0.02em', lineHeight: 1.25 }}>
            Got Questions?{' '}
            <span style={{ background: 'linear-gradient(135deg, #60a5fa 0%, #22d3ee 50%, #a78bfa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
              We Have Answers
            </span>
          </h2>
          <p style={{ color: '#94A3B8', fontSize: '1.08rem', maxWidth: '640px', margin: '0 auto', lineHeight: 1.75 }}>
            Everything you need to know about working with Webotixs, our development workflow, and technology capabilities.
          </p>
        </ScrollReveal>

        {/* FAQ List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {faqList.map((faq, index) => {
            const isOpen = openIndex === index
            return (
              <ScrollReveal key={index} delay={index * 0.08}>
                <div className={`faq-card${isOpen ? ' faq-open' : ''}`}>
                  <button
                    type="button"
                    onClick={() => toggle(index)}
                    aria-expanded={isOpen}
                    aria-controls={`faq-answer-${index}`}
                    style={{ width: '100%', padding: '24px 28px', textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', background: 'none', border: 'none', cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <div className={`faq-icon-box${isOpen ? ' faq-icon-open' : ''}`}>
                        {isOpen ? <Sparkles size={18} /> : <HelpCircle size={18} />}
                      </div>
                      <span style={{ fontWeight: 700, fontSize: '1.05rem', color: '#ffffff', lineHeight: 1.4 }}>
                        {faq.question}
                      </span>
                    </div>
                    <div className={`faq-chevron${isOpen ? ' faq-chevron-open' : ''}`}>
                      <ChevronDown size={18} />
                    </div>
                  </button>

                  <div
                    id={`faq-answer-${index}`}
                    className={`faq-answer${isOpen ? ' faq-answer-open' : ''}`}
                  >
                    <div className="faq-answer-inner">
                      <p style={{ color: '#94A3B8', fontSize: '0.925rem', lineHeight: 1.85, paddingLeft: '86px', paddingRight: '28px', paddingBottom: '24px' }}>
                        {faq.answer}
                      </p>
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            )
          })}
        </div>
      </div>
    </section>
  )
}
