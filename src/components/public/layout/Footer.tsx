import Link from 'next/link'
import { Github, Linkedin, Twitter, Instagram, Facebook, Mail, Phone, MapPin, ArrowRight } from 'lucide-react'

const footerLinks = {
  services: [
    { label: 'Web Design & Development', href: '/services/web-design-development' },
    { label: 'Mobile App Development', href: '/services/mobile-app-development' },
    { label: 'Brand Identity & Design', href: '/services/brand-identity-design' },
    { label: 'E-Commerce Solutions', href: '/services/ecommerce-solutions' },
    { label: 'SEO & Digital Marketing', href: '/services/seo-digital-marketing' },
  ],
  company: [
    { label: 'About Us', href: '/about' },
    { label: 'Portfolio', href: '/portfolio' },
    { label: 'Industries', href: '/industries' },
    { label: 'Team', href: '/team' },
    { label: 'Careers', href: '/careers' },
    { label: 'Blog', href: '/blog' },
  ],
  legal: [
    { label: 'Privacy Policy', href: '/privacy-policy' },
    { label: 'Terms of Service', href: '/terms' },
    { label: 'Cookie Policy', href: '/cookie-policy' },
  ],
}

export default function Footer() {
  return (
    <footer className="relative bg-background-secondary border-t border-border/50 overflow-hidden" suppressHydrationWarning>
      {/* Gradient mesh */}
      <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* CTA Banner */}
        <div className="py-16 border-b border-border/50">
          <div className="glass rounded-3xl p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary-from/10 to-primary-to/10 rounded-3xl" />
            <div className="relative">
              <h2 className="font-display text-3xl md:text-4xl font-bold text-text-white mb-4">
                Ready to Build Something{' '}
                <span className="gradient-text">Amazing?</span>
              </h2>
              <p className="text-text-gray text-lg mb-8 max-w-2xl mx-auto">
                Let&apos;s transform your vision into a stunning digital experience that drives real business results.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link
                  href="/contact"
                  className="px-8 py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl hover:shadow-glow-md transition-all duration-300 hover:scale-105 flex items-center gap-2"
                >
                  Start Your Project <ArrowRight size={18} />
                </Link>
                <Link
                  href="/portfolio"
                  className="px-8 py-4 glass border border-border text-text-white font-semibold rounded-2xl hover:border-primary/50 transition-all duration-300"
                >
                  View Our Work
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Main footer */}
        <div className="py-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-12">
          {/* Brand */}
          <div className="lg:col-span-2">
            <Link href="/" className="flex items-center gap-2 mb-6 group">
              <div className="relative w-8 h-8">
                <div className="absolute inset-0 bg-gradient-to-br from-primary-from to-primary-to rounded-lg" />
                <div className="absolute inset-0.5 bg-background-secondary rounded-md flex items-center justify-center">
                  <span className="text-xs font-bold gradient-text">W</span>
                </div>
              </div>
              <span className="font-display font-bold text-xl text-text-white">Webotixs</span>
            </Link>
            <p className="text-text-gray text-sm leading-relaxed mb-6 max-w-sm">
              We are a premium digital agency crafting beautiful websites, mobile apps, and brand identities that help businesses grow in the digital age.
            </p>
            <div className="flex flex-col gap-3 mb-8">
              <a href="mailto:hello@webotixs.com" className="flex items-center gap-2 text-text-gray hover:text-primary transition-colors text-sm">
                <Mail size={14} /> hello@webotixs.com
              </a>
              <a href="tel:+1234567890" className="flex items-center gap-2 text-text-gray hover:text-primary transition-colors text-sm">
                <Phone size={14} /> +1 (234) 567-890
              </a>
              <span className="flex items-center gap-2 text-text-gray text-sm">
                <MapPin size={14} /> Dubai, UAE · Remote Worldwide
              </span>
            </div>
            {/* Social */}
            <div className="flex items-center gap-3">
              {[
                { icon: Twitter, href: '#', label: 'Twitter' },
                { icon: Linkedin, href: '#', label: 'LinkedIn' },
                { icon: Instagram, href: '#', label: 'Instagram' },
                { icon: Github, href: '#', label: 'GitHub' },
                { icon: Facebook, href: '#', label: 'Facebook' },
              ].map(({ icon: Icon, href, label }) => (
                <a
                  key={label}
                  href={href}
                  aria-label={label}
                  className="w-9 h-9 rounded-xl glass border border-border flex items-center justify-center text-text-gray hover:text-primary hover:border-primary/50 transition-all duration-200"
                >
                  <Icon size={15} />
                </a>
              ))}
            </div>
          </div>

          {/* Services */}
          <div>
            <h3 className="font-display font-semibold text-text-white mb-5 text-sm uppercase tracking-wider">Services</h3>
            <ul className="flex flex-col gap-3">
              {footerLinks.services.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="text-text-gray hover:text-primary text-sm transition-colors animated-underline">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company */}
          <div>
            <h3 className="font-display font-semibold text-text-white mb-5 text-sm uppercase tracking-wider">Company</h3>
            <ul className="flex flex-col gap-3">
              {footerLinks.company.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="text-text-gray hover:text-primary text-sm transition-colors animated-underline">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Newsletter */}
          <div>
            <h3 className="font-display font-semibold text-text-white mb-5 text-sm uppercase tracking-wider">Newsletter</h3>
            <p className="text-text-gray text-sm mb-4">Get weekly insights on web design, tech, and digital growth.</p>
            <form className="flex flex-col gap-2">
              <input
                type="email"
                placeholder="your@email.com"
                className="w-full px-4 py-2.5 bg-background-card border border-border rounded-xl text-text-white text-sm placeholder:text-text-gray/50 focus:outline-none focus:border-primary/50 transition-colors"
              />
              <button
                type="submit"
                className="w-full py-2.5 bg-gradient-to-r from-primary-from to-primary-to text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all"
              >
                Subscribe
              </button>
            </form>
            <div className="mt-6">
              <h4 className="font-semibold text-text-white mb-3 text-sm">Legal</h4>
              <ul className="flex flex-col gap-2">
                {footerLinks.legal.map((link) => (
                  <li key={link.href}>
                    <Link href={link.href} className="text-text-gray hover:text-primary text-xs transition-colors">
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div className="py-6 border-t border-border/50 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-text-gray text-xs">
            &copy; {new Date().getFullYear()} Webotixs. All rights reserved.
          </p>
          <p className="text-text-gray text-xs">
            Designed &amp; Built with <span className="text-primary">❤</span> by Webotixs
          </p>
        </div>
      </div>
    </footer>
  )
}
