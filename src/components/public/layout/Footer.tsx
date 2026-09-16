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
  locations: [
    { label: 'Dallas, TX', href: '/locations/dallas-tx' },
    { label: 'Las Vegas, NV', href: '/locations/las-vegas-nv' },
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
                  aria-label="Start Your Web Design Project with Webotixs"
                  className="px-8 py-4 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl hover:shadow-glow-md transition-all duration-300 hover:scale-105 flex items-center gap-2"
                >
                  Start Your Project <ArrowRight size={18} />
                </Link>
                <Link
                  href="/portfolio"
                  aria-label="View Webotixs Portfolio Case Studies"
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
            <Link href="/" aria-label="Webotixs Homepage" className="flex items-center gap-2 mb-6 group">
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
              <a href="mailto:info@webotixs.com" aria-label="Send email to info@webotixs.com" className="flex items-center gap-2 text-text-gray hover:text-primary transition-colors text-sm">
                <Mail size={14} /> info@webotixs.com
              </a>
              <a href="tel:+12089055973" aria-label="Call US office +1 (208) 905-5973" className="flex items-center gap-2 text-text-gray hover:text-primary transition-colors text-sm">
                <Phone size={14} /> +1 (208) 905-5973
              </a>
              <a href="tel:+923092715559" aria-label="Call Pakistan office +92 309 2715559" className="flex items-center gap-2 text-text-gray hover:text-primary transition-colors text-sm">
                <Phone size={14} /> +92 309 2715559
              </a>
              <span className="flex items-start gap-2 text-text-gray text-sm leading-relaxed">
                <MapPin size={16} className="mt-0.5 flex-shrink-0" /> Mz floor, Al-Qadir Heights, Kalma Chowk Flyover، Babar Block Garden Town, Lahore, 54000, Pakistan
              </span>
            </div>
            {/* Social & Review Platforms */}
            <div className="flex flex-wrap items-center gap-3">
              {[
                { type: 'icon', icon: Facebook, href: 'https://www.facebook.com/profile.php?id=61585230680319', label: 'Facebook', hoverColor: 'hover:text-blue-500 hover:border-blue-500/50' },
                { type: 'icon', icon: Linkedin, href: 'https://www.linkedin.com/in/huzaifa-rao-827444325/', label: 'LinkedIn', hoverColor: 'hover:text-blue-400 hover:border-blue-400/50' },
                { type: 'icon', icon: Instagram, href: 'https://www.instagram.com/webotixs?igsh=dzRhNjV6cGptNXZp', label: 'Instagram', hoverColor: 'hover:text-pink-500 hover:border-pink-500/50' },
                {
                  type: 'svg',
                  href: 'https://clutch.co/profile/webotixs',
                  label: 'Clutch Review Profile',
                  hoverColor: 'hover:text-red-500 hover:border-red-500/50',
                  svg: (
                    <svg width={15} height={15} viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                    </svg>
                  )
                },
                {
                  type: 'svg',
                  href: 'https://www.trustpilot.com/review/webotixs.com',
                  label: 'Trustpilot Review Profile',
                  hoverColor: 'hover:text-emerald-400 hover:border-emerald-400/50',
                  svg: (
                    <svg width={15} height={15} viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
                    </svg>
                  )
                },
                {
                  type: 'svg',
                  href: 'https://g.page/r/Cbbz0-7qGKZVEBM/review',
                  label: 'Google Business Reviews',
                  hoverColor: 'hover:text-blue-400 hover:border-blue-400/50',
                  svg: (
                    <svg width={15} height={15} viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 15.96 0 12.48 0 5.8 0 0 5.4 0 12s5.8 12 12.48 12c3.6 0 6.32-1.187 8.44-3.413 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z" />
                    </svg>
                  )
                },
              ].map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={item.label}
                  title={item.label}
                  className={`w-9 h-9 rounded-xl glass border border-border flex items-center justify-center text-text-gray ${item.hoverColor} transition-all duration-200 hover:scale-110 shadow-sm`}
                >
                  {item.type === 'icon' && item.icon ? <item.icon size={15} /> : item.svg}
                  <span className="sr-only">{item.label}</span>
                </a>
              ))}
            </div>
          </div>

          {/* Services */}
          <div>
            <p className="font-display font-semibold text-text-white mb-5 text-sm uppercase tracking-wider">Services</p>
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
            <p className="font-display font-semibold text-text-white mb-5 text-sm uppercase tracking-wider">Company</p>
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
            <p className="font-display font-semibold text-text-white mb-5 text-sm uppercase tracking-wider">Newsletter</p>
            <p className="text-text-gray text-sm mb-4">Get weekly insights on web design, tech, and digital growth.</p>
            <form className="flex flex-col gap-2">
              <input
                type="email"
                placeholder="your@email.com"
                aria-label="Email address for newsletter subscription"
                className="w-full px-4 py-2.5 bg-background-card border border-border rounded-xl text-text-white text-sm placeholder:text-text-gray/50 focus:outline-none focus:border-primary/50 transition-colors"
              />
              <button
                type="submit"
                aria-label="Subscribe to Webotixs weekly newsletter"
                className="w-full py-2.5 bg-gradient-to-r from-primary-from to-primary-to text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all"
              >
                Subscribe
              </button>
            </form>
            <div className="mt-6 grid grid-cols-2 gap-4">
              <div>
                <p className="font-semibold text-text-white mb-3 text-sm">Legal</p>
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
              <div>
                <p className="font-semibold text-text-white mb-3 text-sm">Locations</p>
                <ul className="flex flex-col gap-2">
                  {footerLinks.locations.map((link) => (
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
