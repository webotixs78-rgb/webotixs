'use client'

import { useState } from 'react'
import { Mail, Phone, MapPin, Send, Loader2, MessageSquare, Briefcase, User } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { contactSchema, type ContactFormData } from '@/lib/validations/contact'
import ScrollReveal from '@/components/animations/ScrollReveal'

export default function ContactPage() {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitSuccess, setSubmitSuccess] = useState(false)

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
      company: '',
      service: '',
      budget: '',
      message: '',
    },
  })

  const onSubmit = async (data: ContactFormData) => {
    setIsSubmitting(true)
    try {
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      const json = await res.json()
      console.log('[Contact Submission Status]:', json.emailsSent, json.diagnostics)
      if (!res.ok) {
        throw new Error(json.error || 'Failed to send message')
      }

      if (json.lead || json.inquiry) {
        try {
          const leads = JSON.parse(localStorage.getItem('webotixs_crm_leads') || '[]')
          if (Array.isArray(leads)) {
            leads.unshift(json.lead || json.inquiry)
            localStorage.setItem('webotixs_crm_leads', JSON.stringify(leads))
          }
          const inquiries = JSON.parse(localStorage.getItem('webotixs_contact_inquiries') || '[]')
          if (Array.isArray(inquiries)) {
            inquiries.unshift(json.inquiry || json.lead)
            localStorage.setItem('webotixs_contact_inquiries', JSON.stringify(inquiries))
          }
        } catch {}
      }
      if (json.notification) {
        try {
          const notifs = JSON.parse(localStorage.getItem('webotixs_crm_notifications') || '[]')
          if (Array.isArray(notifs)) {
            notifs.unshift(json.notification)
            localStorage.setItem('webotixs_crm_notifications', JSON.stringify(notifs))
          }
        } catch {}
      }
      window.dispatchEvent(new Event('storage'))

      setSubmitSuccess(true)
      reset()
    } catch (e) {
      console.error(e)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="pt-24 bg-background min-h-screen">
      <section className="relative py-16 md:py-24 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-20 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 lg:gap-16 items-start">
            {/* Contact Info Sidebar */}
            <div className="lg:col-span-2 space-y-8">
              <ScrollReveal>
                <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
                  <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
                  <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Get in Touch</span>
                </div>
                <h1 className="font-display text-4xl sm:text-5xl font-bold text-text-white mb-6">
                  Let&apos;s Build <br /> Something <span className="gradient-text">Great</span>
                </h1>
                <p className="text-text-gray text-sm md:text-base leading-relaxed max-w-sm">
                  Have an idea, project, or need some support? Reach out using the form or direct channels. We respond within 24 hours.
                </p>
              </ScrollReveal>

              <ScrollReveal className="space-y-6">
                {/* Mail */}
                <div className="flex items-start gap-4">
                  <div className="w-11 h-11 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary flex-shrink-0">
                    <Mail size={16} />
                  </div>
                  <div>
                    <h3 className="text-xs uppercase text-text-gray font-bold tracking-wider mb-1">Email Us</h3>
                    <a href="mailto:info@webotixs.com" className="text-text-white hover:text-primary transition-colors text-sm font-semibold">
                      info@webotixs.com
                    </a>
                  </div>
                </div>

                {/* Phone */}
                <div className="flex items-start gap-4">
                  <div className="w-11 h-11 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary flex-shrink-0">
                    <Phone size={16} />
                  </div>
                  <div>
                    <h3 className="text-xs uppercase text-text-gray font-bold tracking-wider mb-1">Call Us</h3>
                    <div className="space-y-1">
                      <a href="tel:+12089055973" className="block text-text-white hover:text-primary transition-colors text-sm font-semibold">
                        +1 (208) 905-5973
                      </a>
                      <a href="tel:+923092715559" className="block text-text-white hover:text-primary transition-colors text-sm font-semibold">
                        +92 309 2715559
                      </a>
                    </div>
                  </div>
                </div>

                {/* Map */}
                <div className="flex items-start gap-4">
                  <div className="w-11 h-11 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary flex-shrink-0">
                    <MapPin size={16} />
                  </div>
                  <div>
                    <h3 className="text-xs uppercase text-text-gray font-bold tracking-wider mb-1">Office Location</h3>
                    <span className="text-text-white text-sm font-semibold leading-relaxed block">
                      Mz floor, Al-Qadir Heights, Kalma Chowk Flyover، Babar Block Garden Town, Lahore, 54000, Pakistan
                    </span>
                  </div>
                </div>
              </ScrollReveal>
            </div>

            {/* Contact Form */}
            <div className="lg:col-span-3">
              <ScrollReveal className="glass rounded-3xl p-8 border border-border/60">
                {submitSuccess ? (
                  <div className="text-center py-12 space-y-4">
                    <div className="w-16 h-16 rounded-full bg-success/15 border border-success/30 flex items-center justify-center mx-auto text-success">
                      ✓
                    </div>
                    <h2 className="font-display text-2xl font-bold text-text-white">Message Sent Successfully!</h2>
                    <p className="text-text-gray text-sm max-w-sm mx-auto">
                      Thank You! Your project inquiry has been submitted successfully. A confirmation email has been sent to your inbox. Our team will review your requirements and contact you within 24 business hours.
                    </p>
                    <button
                      onClick={() => setSubmitSuccess(false)}
                      className="px-6 py-2.5 bg-background border border-border rounded-xl text-xs font-semibold text-text-white hover:text-primary transition-colors"
                    >
                      Send Another Message
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                      {/* Name */}
                      <div className="space-y-2">
                        <label htmlFor="name" className="text-xs font-semibold text-text-gray flex items-center gap-1.5">
                          <User size={13} /> Full Name *
                        </label>
                        <input
                          id="name"
                          type="text"
                          placeholder="John Doe"
                          {...register('name')}
                          className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/50 transition-colors"
                        />
                        {errors.name && <p className="text-xs text-red-500 font-medium">{errors.name.message}</p>}
                      </div>

                      {/* Email */}
                      <div className="space-y-2">
                        <label htmlFor="email" className="text-xs font-semibold text-text-gray flex items-center gap-1.5">
                          <Mail size={13} /> Email Address *
                        </label>
                        <input
                          id="email"
                          type="email"
                          placeholder="john@example.com"
                          {...register('email')}
                          className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/50 transition-colors"
                        />
                        {errors.email && <p className="text-xs text-red-500 font-medium">{errors.email.message}</p>}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                      {/* Phone */}
                      <div className="space-y-2">
                        <label htmlFor="phone" className="text-xs font-semibold text-text-gray flex items-center gap-1.5">
                          <Phone size={13} /> Phone Number (Optional)
                        </label>
                        <input
                          id="phone"
                          type="text"
                          placeholder="+1 (234) 567-890"
                          {...register('phone')}
                          className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/50 transition-colors"
                        />
                      </div>

                      {/* Company */}
                      <div className="space-y-2">
                        <label htmlFor="company" className="text-xs font-semibold text-text-gray flex items-center gap-1.5">
                          <Briefcase size={13} /> Company / Organization
                        </label>
                        <input
                          id="company"
                          type="text"
                          placeholder="Acme Corp"
                          {...register('company')}
                          className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/50 transition-colors"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                      {/* Service Choice */}
                      <div className="space-y-2">
                        <label htmlFor="service" className="text-xs font-semibold text-text-gray">Service Type</label>
                        <select
                          id="service"
                          {...register('service')}
                          className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm focus:outline-none focus:border-primary/50 transition-colors"
                        >
                          <option value="">Select a service...</option>
                          <option value="web-development">Web Design &amp; Development</option>
                          <option value="mobile-apps">Mobile App Development</option>
                          <option value="brand-identity">Brand Identity &amp; Design</option>
                          <option value="ecommerce">E-Commerce Solutions</option>
                          <option value="seo">SEO &amp; Digital Marketing</option>
                        </select>
                      </div>

                      {/* Budget Choice */}
                      <div className="space-y-2">
                        <label htmlFor="budget" className="text-xs font-semibold text-text-gray">Project Budget</label>
                        <select
                          id="budget"
                          {...register('budget')}
                          className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm focus:outline-none focus:border-primary/50 transition-colors"
                        >
                          <option value="">Select a budget range...</option>
                          <option value="5k-10k">$5,000 - $10,000</option>
                          <option value="10k-25k">$10,000 - $25,000</option>
                          <option value="25k-50k">$25,000 - $50,000</option>
                          <option value="50k-plus">$50,000+</option>
                        </select>
                      </div>
                    </div>

                    {/* Message */}
                    <div className="space-y-2">
                      <label htmlFor="message" className="text-xs font-semibold text-text-gray flex items-center gap-1.5">
                        <MessageSquare size={13} /> Project Details *
                      </label>
                      <textarea
                        id="message"
                        rows={5}
                        placeholder="Tell us about your project, goals, timeline, and requirements..."
                        {...register('message')}
                        className="w-full px-4 py-3 bg-background border border-border rounded-2xl text-text-white text-sm placeholder:text-text-gray/40 focus:outline-none focus:border-primary/50 transition-colors resize-none"
                      />
                      {errors.message && <p className="text-xs text-red-500 font-medium">{errors.message.message}</p>}
                    </div>

                    {/* Submit */}
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="flex items-center justify-center gap-2 w-full py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-102 disabled:opacity-50 disabled:pointer-events-none"
                    >
                      {isSubmitting ? (
                        <>
                          <Loader2 size={16} className="animate-spin" /> Sending Details...
                        </>
                      ) : (
                        <>
                          <Send size={16} /> Send Message
                        </>
                      )}
                    </button>
                  </form>
                )}
              </ScrollReveal>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
