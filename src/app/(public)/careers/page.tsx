import Link from 'next/link'
import { Briefcase, MapPin, Clock, ArrowRight, ShieldCheck, Heart, Sparkles } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'

export const metadata = {
  title: 'Careers',
  description: 'Join the team at Webotixs. Work on premium React websites, mobile apps, and elite server infrastructures.',
}

const perks = [
  {
    icon: Heart,
    title: 'Health & Wellness',
    description: 'Comprehensive private medical coverage, wellness programs, and gym memberships allowance.',
  },
  {
    icon: Sparkles,
    title: 'Growth & Education',
    description: 'Annual training budget, book allowance, paid certificates, and conference tickets.',
  },
  {
    icon: ShieldCheck,
    title: 'Flexible & Hybrid',
    description: 'Choose your hours, work from home or our premium Dubai office, with high-quality setup gear provided.',
  },
]

const jobs = [
  {
    id: 'senior-frontend-engineer',
    title: 'Senior Frontend Engineer',
    department: 'Engineering',
    location: 'Dubai / Remote',
    type: 'Full-time',
    salary: '$80,000 - $110,000',
  },
  {
    id: 'lead-ux-designer',
    title: 'Lead UX/UI Designer',
    department: 'Design',
    location: 'Remote',
    type: 'Full-time',
    salary: '$70,000 - $95,000',
  },
  {
    id: 'devops-architect',
    title: 'Cloud & DevOps Architect',
    department: 'Operations',
    location: 'Dubai / Hybrid',
    type: 'Full-time',
    salary: '$90,000 - $130,000',
  },
]

export default function CareersPage() {
  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-primary-from rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Join Us</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Build the Future <br /> of the <span className="gradient-text">Web</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              We are looking for exceptional developers, designers, and marketers to help us craft the next generation of premium applications.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Perks Section */}
      <section className="py-16 md:py-24 border-t border-border/40 bg-background-secondary relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center mb-16">
            <h2 className="font-display text-3xl font-bold text-text-white">Why Join Webotixs?</h2>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {perks.map((perk, i) => {
              const Icon = perk.icon
              return (
                <ScrollReveal key={perk.title} delay={i * 0.1}>
                  <div className="bg-background-card border border-border/60 rounded-3xl p-8 hover:border-primary/45 transition-colors duration-300">
                    <div className="w-12 h-12 bg-primary/10 rounded-2xl flex items-center justify-center mb-6 text-primary border border-primary/20">
                      <Icon size={20} />
                    </div>
                    <h3 className="font-display text-lg font-bold text-text-white mb-3">{perk.title}</h3>
                    <p className="text-text-gray text-sm leading-relaxed">{perk.description}</p>
                  </div>
                </ScrollReveal>
              )
            })}
          </div>
        </div>
      </section>

      {/* Openings Section */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center mb-16">
            <h2 className="font-display text-3xl font-bold text-text-white">Current Openings</h2>
          </ScrollReveal>

          <div className="space-y-4">
            {jobs.map((job, i) => (
              <ScrollReveal key={job.id} delay={i * 0.1}>
                <div className="glass border border-border/60 rounded-3xl p-6 md:p-8 flex flex-col md:flex-row md:items-center md:justify-between gap-6 hover:border-primary/40 transition-colors">
                  <div className="space-y-2">
                    <h3 className="font-display text-xl font-bold text-text-white">{job.title}</h3>
                    <div className="flex flex-wrap items-center gap-4 text-text-gray text-xs">
                      <span className="flex items-center gap-1">
                        <Briefcase size={12} />
                        {job.department}
                      </span>
                      <span className="flex items-center gap-1">
                        <MapPin size={12} />
                        {job.location}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock size={12} />
                        {job.type}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between md:justify-end gap-6">
                    <span className="text-primary text-sm font-semibold">{job.salary}</span>
                    <Link
                      href="/contact"
                      className="px-5 py-3 bg-gradient-to-r from-primary-from to-primary-to text-white text-xs font-bold rounded-xl flex items-center gap-1.5 hover:shadow-glow-sm transition-all"
                    >
                      Apply Now <ArrowRight size={14} />
                    </Link>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
