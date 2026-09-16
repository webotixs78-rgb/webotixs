import { ArrowRight, CheckCircle2, Shield, Users, Lightbulb, Trophy } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import AnimatedCounter from '@/components/animations/AnimatedCounter'

export const metadata = {
  title: 'About Us',
  description: 'Learn about Webotixs, our mission, vision, values, and the team driving our digital innovations.',
}

const values = [
  {
    icon: Lightbulb,
    title: 'Innovation',
    description: 'We push the boundaries of what is possible, staying ahead of digital trends to deliver state-of-the-art solutions.',
  },
  {
    icon: Shield,
    title: 'Integrity',
    description: 'We believe in absolute transparency, honest pricing, and high ethical standards in all client relationships.',
  },
  {
    icon: Users,
    title: 'Collaboration',
    description: 'We work closely as partners with our clients, treating their business goals as our very own.',
  },
  {
    icon: Trophy,
    title: 'Excellence',
    description: 'We hold ourselves to world-class standards, refusing to settle for mediocrity in design, code, or support.',
  },
]

const milestones = [
  { year: '2018', title: 'Agency Founded', description: 'Webotixs opened its doors in Lahore, Pakistan with a core team of three innovators.' },
  { year: '2020', title: 'Global Scaling', description: 'Expanded remote operations globally, serving clients across North America, Europe, and the Middle East.' },
  { year: '2022', title: '50+ Team Milestone', description: 'Grew our core staff to 50+ designers, full-stack engineers, and marketing experts.' },
  { year: '2024', title: 'Enterprise Partnerships', description: 'Established strategic partnerships with Fortune 500 brands and high-growth fintech platforms.' },
]

export default function AboutPage() {
  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Our Story</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              Crafting the <span className="gradient-text">Future</span> <br /> of Digital Innovation
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              We are a team of passionate creators, engineers, and digital strategists who believe that stunning design and flawless technology are key to business growth.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Story & Philosophy */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <ScrollReveal direction="left">
              <h2 className="font-display text-3xl md:text-4xl font-bold text-text-white mb-6">
                Redefining the Agency Experience
              </h2>
              <div className="space-y-4 text-text-gray">
                <p>
                  Founded in 2018, Webotixs set out to bridge the gap between boutique design agencies and large-scale technical integrators. We realized that companies shouldn&apos;t have to compromise beautiful aesthetics for high-performance software engineering.
                </p>
                <p>
                  Today, we build software products that feel premium, perform at lightning speeds, and scale effortlessly. From dynamic startup web apps to complex database migrations, we make it our mission to deliver state-of-the-art work.
                </p>
              </div>

              <div className="mt-8 grid grid-cols-2 gap-6">
                <div>
                  <div className="font-display text-4xl font-bold gradient-text mb-1">
                    <AnimatedCounter end={200} suffix="+" />
                  </div>
                  <div className="text-text-gray text-sm">Projects successfully deployed</div>
                </div>
                <div>
                  <div className="font-display text-4xl font-bold gradient-text mb-1">
                    <AnimatedCounter end={98} suffix="%" />
                  </div>
                  <div className="text-text-gray text-sm">Client retention and satisfaction</div>
                </div>
              </div>
            </ScrollReveal>

            <ScrollReveal direction="right" className="relative lg:h-[450px] bg-gradient-to-br from-background-section to-background-secondary rounded-3xl border border-border/80 flex items-center justify-center p-8 overflow-hidden">
              <div className="absolute inset-0 bg-primary/5 opacity-50 pointer-events-none" />
              <div className="relative text-center">
                <div className="w-20 h-20 bg-gradient-to-tr from-primary-from to-primary-to rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-glow-sm">
                  <CheckCircle2 size={36} className="text-text-white" />
                </div>
                <h3 className="font-display text-2xl font-bold text-text-white mb-3">Our Core Promise</h3>
                <p className="text-text-gray text-sm max-w-sm mx-auto leading-relaxed">
                  We don&apos;t build cookie-cutter templates. Everything we ship is uniquely customized, rigorously tested, and fully optimized for conversions.
                </p>
              </div>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Core Values */}
      <section className="py-16 md:py-24 bg-background-secondary border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center mb-16">
            <h2 className="font-display text-3xl md:text-4xl font-bold text-text-white">
              The Values That <span className="gradient-text">Drive Us</span>
            </h2>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((val, i) => {
              const Icon = val.icon
              return (
                <ScrollReveal key={val.title} delay={i * 0.1}>
                  <div className="h-full bg-background-card border border-border/60 rounded-3xl p-6 hover:border-primary/40 transition-colors duration-300">
                    <div className="w-12 h-12 bg-primary/10 rounded-2xl flex items-center justify-center mb-5 text-primary">
                      <Icon size={22} />
                    </div>
                    <h3 className="font-display text-lg font-bold text-text-white mb-2">{val.title}</h3>
                    <p className="text-text-gray text-sm leading-relaxed">{val.description}</p>
                  </div>
                </ScrollReveal>
              )
            })}
          </div>
        </div>
      </section>

      {/* Timeline/History */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal className="text-center mb-16">
            <h2 className="font-display text-3xl md:text-4xl font-bold text-text-white">
              Our Journey <span className="gradient-text">So Far</span>
            </h2>
          </ScrollReveal>

          <div className="relative border-l border-border/60 ml-4 md:ml-6 space-y-12">
            {milestones.map((m, i) => (
              <ScrollReveal key={m.year} delay={i * 0.1} className="relative pl-8 md:pl-10">
                <div className="absolute -left-[9px] top-1.5 w-4.5 h-4.5 rounded-full bg-primary border-4 border-background" />
                <div className="font-display text-xl font-bold text-primary mb-1">{m.year}</div>
                <h3 className="font-display text-lg font-bold text-text-white mb-2">{m.title}</h3>
                <p className="text-text-gray text-sm leading-relaxed">{m.description}</p>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
