'use client'

import React from 'react'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

interface TechLogo {
  name: string
  category: string
  color: string
  svg: React.ReactNode
}

const techLogos: TechLogo[] = [
  {
    name: 'WordPress',
    category: 'CMS & Web Development',
    color: '#21759B',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M12.158 12.786l-2.698 7.84c.806.236 1.657.364 2.54.364 2.36 0 4.498-.915 6.096-2.417l-1.637-4.48-4.301-1.307zm-7.619 1.13c0 2.92 1.489 5.492 3.754 7.027l-3.328-9.135c-.276.671-.426 1.393-.426 2.108zm14.399-5.111c0-1.286-.462-2.176-.859-2.868-.528-.859-.991-1.585-.991-2.443 0-.958.726-1.849 1.749-1.849.046 0 .092.003.137.006A9.92 9.92 0 0 0 12 0C5.373 0 0 5.373 0 12c0 2.215.6 4.29 1.644 6.071l4.981-13.655c.462-1.287 1.056-1.585 1.947-1.585.826 0 1.453.462 1.453 1.387 0 .825-.495 1.783-1.023 2.774-.693 1.287-1.386 2.707-1.386 4.358 0 2.377 1.386 4.159 3.235 4.159 1.849 0 2.707-1.254 2.707-2.74 0-1.42-.693-2.673-1.386-3.927-.528-.991-1.023-1.949-1.023-2.774 0-.925.627-1.387 1.453-1.387.891 0 1.485.298 1.947 1.585l2.25 6.177 1.603 4.397c1.378-1.597 2.211-3.674 2.211-5.945 0-.414-.029-.821-.086-1.22z"/>
      </svg>
    ),
  },
  {
    name: 'Netlify',
    category: 'Cloud Deployment',
    color: '#00C7B7',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M6.41 11.23l2.87-2.87-2.87-2.87-2.87 2.87 2.87 2.87zm5.59-5.59l2.87-2.87-2.87-2.77L9.13 2.77l2.87 2.87zm5.59 5.59l2.87-2.87-2.87-2.87-2.87 2.87 2.87 2.87zm-5.59 5.59l2.87-2.87-2.87-2.87-2.87 2.87 2.87 2.87zM.77 12l2.87 2.87 2.87-2.87L3.64 9.13.77 12zm22.46 0l-2.87-2.87-2.87 2.87 2.87 2.87 2.87-2.87z"/>
      </svg>
    ),
  },
  {
    name: 'Next.js 16',
    category: 'React Enterprise Framework',
    color: '#3B82F6',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M18.665 21.978C16.758 23.255 14.465 24 12 24 5.373 24 0 18.627 0 12S5.373 0 12 0s12 5.373 12 12c0 3.584-1.574 6.801-4.067 9.001l-10.9-14.072H6.75v10.14h2.25v-6.981l8.665 11.89zm-2.915-4.908l2.25 3.085V6.93h-2.25v10.14z"/>
      </svg>
    ),
  },
  {
    name: 'React 19',
    category: 'UI Library',
    color: '#61DAFB',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm0-7.5c-4.97 0-9 2.01-9 4.5 0 1.93 2.47 3.58 6 4.19V8.67C6.06 8.24 4.5 7.15 4.5 6c0-1.66 3.36-3 7.5-3s7.5 1.34 7.5 3c0 1.15-1.56 2.24-4.5 2.67v1.52c3.53-.61 6-2.26 6-4.19 0-2.49-4.03-4.5-9-4.5zM12 22.5c4.97 0 9-2.01 9-4.5 0-1.93-2.47-3.58-6-4.19v1.52c2.94.43 4.5 1.52 4.5 2.67 0 1.66-3.36 3-7.5 3s-7.5-1.34-7.5-3c0-1.15 1.56-2.24 4.5-2.67v-1.52c-3.53.61-6 2.26-6 4.19 0 2.49 4.03 4.5 9 4.5z"/>
      </svg>
    ),
  },
  {
    name: 'Supabase',
    category: 'Backend & Cloud Storage',
    color: '#3ECF8E',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M13.359 1.144L2.643 14.167a.75.75 0 00.58 1.228h7.918l-1.5 7.46a.75.75 0 001.295.632l10.716-13.022a.75.75 0 00-.58-1.228h-7.918l1.5-7.46a.75.75 0 00-1.295-.633z"/>
      </svg>
    ),
  },
  {
    name: 'Tailwind CSS',
    category: 'Modern Design System',
    color: '#38BDF8',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M12.001 4.8c-3.2 0-5.2 1.6-6 4.8 1.2-1.6 2.6-2.2 4.2-1.8.913.228 1.565.89 2.288 1.624C13.666 10.618 15.027 12 18.001 12c3.2 0 5.2-1.6 6-4.8-1.2 1.6-2.6 2.2-4.2 1.8-.913-.228-1.565-.89-2.288-1.624C16.337 6.182 14.976 4.8 12.001 4.8zm-6 7.2c-3.2 0-5.2 1.6-6 4.8 1.2-1.6 2.6-2.2 4.2-1.8.913.228 1.565.89 2.288 1.624C7.666 17.818 9.027 19.2 12.001 19.2c3.2 0 5.2-1.6 6-4.8-1.2 1.6-2.6 2.2-4.2 1.8-.913-.228-1.565-.89-2.288-1.624C10.337 13.382 8.976 12 6.001 12z"/>
      </svg>
    ),
  },
  {
    name: 'Node.js',
    category: 'Backend APIs',
    color: '#5FA04E',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M12 1.732l8.66 5v10l-8.66 5-8.66-5v-10l8.66-5zm0 2.598L5.34 8.165v7.67L12 19.67l6.66-3.835v-7.67L12 4.33z"/>
      </svg>
    ),
  },
  {
    name: 'Vercel',
    category: 'Global Edge Network',
    color: '#FFFFFF',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M24 22.525H0l12-21.05 12 21.05z"/>
      </svg>
    ),
  },
  {
    name: 'Shopify',
    category: 'E-Commerce Storefronts',
    color: '#96BF48',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M15.334 3.253c-.078-.078-.204-.06-.255.034l-1.34 2.457c-.77-1.07-1.92-1.785-3.238-1.785-2.64 0-4.78 2.14-4.78 4.78 0 1.25.48 2.39 1.27 3.24L4.85 19.2c-.06.11.02.24.14.24h13.97c.12 0 .2-.13.14-.24l-3.77-7.22c.79-.85 1.27-1.99 1.27-3.24 0-1.85-.75-3.52-1.96-4.73z"/>
      </svg>
    ),
  },
  {
    name: 'Figma',
    category: 'UI/UX Prototyping',
    color: '#F24E1E',
    svg: (
      <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
        <path d="M8.5 0C6.57 0 5 1.57 5 3.5S6.57 7 8.5 7H12V0H8.5zm7 0H12v7h3.5C17.43 7 19 5.43 19 3.5S17.43 0 15.5 0zM12 8.5v7h3.5c1.93 0 3.5-1.57 3.5-3.5S17.43 8.5 15.5 8.5H12zM8.5 17c-1.93 0-3.5 1.57-3.5 3.5S6.57 24 8.5 24 12 22.43 12 20.5V17H8.5zM5 12c0-1.93 1.57-3.5 3.5-3.5H12v7H8.5C6.57 15.5 5 13.93 5 12z"/>
      </svg>
    ),
  },
]

export default function TechLogoSlider() {
  return (
    <section className="py-12 relative overflow-hidden bg-background-secondary/40 border-y border-border/40">
      {/* Background Lighting Glow */}
      <div className="absolute left-1/4 top-1/2 -translate-y-1/2 w-96 h-32 bg-primary/10 blur-3xl pointer-events-none rounded-full" />
      <div className="absolute right-1/4 top-1/2 -translate-y-1/2 w-96 h-32 bg-cyan-500/10 blur-3xl pointer-events-none rounded-full" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-8 text-center">
        <span className="text-[11px] font-mono font-bold uppercase tracking-widest text-primary bg-primary/10 border border-primary/20 px-3.5 py-1.5 rounded-full inline-block mb-3">
          Technologies & Ecosystem
        </span>
        <p className="text-xl md:text-2xl font-display font-bold text-text-white">
          Powered by Industry-Leading Platforms & Frameworks
        </p>
        <p className="text-xs text-text-gray mt-1 max-w-xl mx-auto">
          We engineer high-performance web applications, custom CMS solutions, and mobile apps built on robust tech stacks.
        </p>
      </div>

      {/* Infinite Marquee Wrapper */}
      <div className="relative w-full overflow-hidden py-4">
        {/* Left & Right Gradient Fades */}
        <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-background-secondary to-transparent z-20 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-background-secondary to-transparent z-20 pointer-events-none" />

        {/* Marquee Track (Duplicated twice for seamless loop) */}
        <div className="animate-marquee flex items-center gap-6">
          {[...techLogos, ...techLogos].map((item, index) => (
            <GlowingGlassCard
              key={`${item.name}-${index}`}
              className="flex-shrink-0 px-5 py-3.5 min-w-[210px] glass hover:border-primary/50 transition-all cursor-pointer group"
            >
              <div className="flex items-center gap-3.5">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center bg-white/5 border border-white/10 group-hover:scale-110 transition-transform shadow-md"
                  style={{ color: item.color }}
                >
                  {item.svg}
                </div>
                <div>
                  <div className="text-sm font-bold text-text-white group-hover:text-primary transition-colors flex items-center gap-1.5">
                    {item.name}
                  </div>
                  <div className="text-[10px] text-text-gray truncate">{item.category}</div>
                </div>
              </div>
            </GlowingGlassCard>
          ))}
        </div>
      </div>
    </section>
  )
}
