'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { contactSchema, type ContactFormData } from '@/lib/validations/contact'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { Send, CheckCircle2, Loader2, Sparkles, Mail, Phone, MessageSquare } from 'lucide-react'

export default function ContactSection() {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<ContactFormData>({
    resolver: zodResolver(contactSchema),
    defaultValues: {
      name: '',
      email: '',
      phone: '',
      service: 'Web Design & Development',
      message: '',
    },
  })

  const onSubmit = async (data: ContactFormData) => {
    setIsSubmitting(true)
    setErrorMsg(null)

    try {
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })

      const json = await res.json()
      if (!res.ok) {
        throw new Error(json.error || 'Failed to submit inquiry')
      }

      setSuccess(true)
      reset()
      setTimeout(() => setSuccess(false), 6000)
    } catch (err: any) {
      setErrorMsg(err.message || 'An unexpected error occurred. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <section className="py-24 bg-background relative overflow-hidden border-t border-border/50" id="contact-section">
      <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-blue-600/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -top-40 -right-40 w-96 h-96 bg-cyan-500/15 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <ScrollReveal className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 glass rounded-full border border-border/60 mb-4">
            <Sparkles size={14} className="text-cyan-400" />
            <span className="text-text-gray text-xs font-semibold uppercase tracking-wider">Start a Conversation</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-text-white mb-4">
            Let’s Build Your <span className="gradient-text">Next Digital Masterpiece</span>
          </h2>
          <p className="text-text-gray text-base sm:text-lg">
            Have a project in mind or need an enterprise digital consultation? Send our engineers a message today.
          </p>
        </ScrollReveal>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          {/* Left Info Panel */}
          <div className="lg:col-span-5 glass border border-border/60 rounded-3xl p-8 sm:p-10 space-y-8">
            <div>
              <h3 className="font-display text-2xl font-bold text-white mb-3">Get in Touch Directly</h3>
              <p className="text-text-gray text-sm leading-relaxed">
                Our global project managers are ready to review your requirements and provide a detailed technical roadmap within 24 hours.
              </p>
            </div>

            <div className="space-y-5 pt-4 border-t border-border/40">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-400">
                  <Mail size={20} />
                </div>
                <div>
                  <div className="text-xs text-text-gray uppercase font-bold">General & New Business</div>
                  <a href="mailto:hello@webotixs.com" className="text-sm font-semibold text-white hover:text-primary transition-colors">
                    hello@webotixs.com
                  </a>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                  <Phone size={20} />
                </div>
                <div>
                  <div className="text-xs text-text-gray uppercase font-bold">Direct Phone / WhatsApp</div>
                  <a href="tel:+1234567890" className="text-sm font-semibold text-white hover:text-primary transition-colors">
                    +1 (234) 567-890
                  </a>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-violet-500/15 border border-violet-500/30 flex items-center justify-center text-violet-400">
                  <MessageSquare size={20} />
                </div>
                <div>
                  <div className="text-xs text-text-gray uppercase font-bold">AI Lead Classification</div>
                  <span className="text-xs text-emerald-400 font-medium">Automatic 60s priority analysis</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Form Panel */}
          <div className="lg:col-span-7 glass border border-border/60 rounded-3xl p-8 sm:p-10">
            {success && (
              <div className="mb-6 p-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-2xl flex items-center gap-3 text-sm font-semibold">
                <CheckCircle2 size={20} className="flex-shrink-0" />
                <span>Thank you! Your inquiry has been submitted and classified. Our team will contact you shortly.</span>
              </div>
            )}

            {errorMsg && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-2xl text-xs font-semibold">
                {errorMsg}
              </div>
            )}

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label htmlFor="name" className="text-xs font-semibold text-text-gray">
                    Your Name *
                  </label>
                  <input
                    id="name"
                    type="text"
                    placeholder="Ahmed Al-Rashid"
                    {...register('name')}
                    className="w-full px-4 py-3 bg-background border border-border rounded-xl text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/60 transition-colors"
                  />
                  {errors.name && <p className="text-xs text-red-400">{errors.name.message}</p>}
                </div>

                <div className="space-y-1.5">
                  <label htmlFor="email" className="text-xs font-semibold text-text-gray">
                    Email Address *
                  </label>
                  <input
                    id="email"
                    type="email"
                    placeholder="ahmed@company.com"
                    {...register('email')}
                    className="w-full px-4 py-3 bg-background border border-border rounded-xl text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/60 transition-colors"
                  />
                  {errors.email && <p className="text-xs text-red-400">{errors.email.message}</p>}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label htmlFor="phone" className="text-xs font-semibold text-text-gray">
                    Phone / WhatsApp (Optional)
                  </label>
                  <input
                    id="phone"
                    type="text"
                    placeholder="+971 50 123 4567"
                    {...register('phone')}
                    className="w-full px-4 py-3 bg-background border border-border rounded-xl text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/60 transition-colors"
                  />
                </div>

                <div className="space-y-1.5">
                  <label htmlFor="service" className="text-xs font-semibold text-text-gray">
                    Service Required *
                  </label>
                  <select
                    id="service"
                    {...register('service')}
                    className="w-full px-4 py-3 bg-background border border-border rounded-xl text-white text-sm focus:outline-none focus:border-primary/60 transition-colors"
                  >
                    <option value="Web Design & Development">Web Design & Development</option>
                    <option value="Mobile App Development">Mobile App Development</option>
                    <option value="Brand Identity & Design">Brand Identity & Design</option>
                    <option value="E-Commerce Solutions">E-Commerce Solutions</option>
                    <option value="SEO & Digital Marketing">SEO & Digital Marketing</option>
                    <option value="Cloud & DevOps">Cloud & DevOps / Custom Software</option>
                  </select>
                </div>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="message" className="text-xs font-semibold text-text-gray">
                  Project Brief & Requirements *
                </label>
                <textarea
                  id="message"
                  rows={4}
                  placeholder="Tell us about your project goals, target audience, and expected timeline..."
                  {...register('message')}
                  className="w-full px-4 py-3 bg-background border border-border rounded-xl text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/60 transition-colors resize-none"
                />
                {errors.message && <p className="text-xs text-red-400">{errors.message.message}</p>}
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-bold rounded-xl hover:shadow-glow-md transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 size={18} className="animate-spin" /> Submitting Inquiry & AI Analysis...
                  </>
                ) : (
                  <>
                    <Send size={18} /> Submit Project Inquiry
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </section>
  )
}
