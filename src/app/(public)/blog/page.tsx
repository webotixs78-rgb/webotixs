import Link from 'next/link'
import { Calendar, User, Clock, ArrowRight } from 'lucide-react'
import ScrollReveal from '@/components/animations/ScrollReveal'
import { getCMSData } from '@/lib/data/cms'
import { formatDate } from '@/lib/utils'

export const dynamic = 'force-dynamic'

export const metadata = {
  title: 'Blog & Articles',
  description: 'Read the latest thoughts, tutorials, and guidelines on web development, design, and digital business scaling from the Webotixs team.',
}

export default async function BlogPage() {
  const posts: any[] = await getCMSData('blogs')

  return (
    <div className="pt-24 bg-background">
      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden">
        <div className="absolute inset-0 mesh-gradient opacity-30 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <ScrollReveal>
            <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/60 mb-5">
              <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse" />
              <span className="text-text-gray text-xs font-medium uppercase tracking-wider">Insights & News</span>
            </div>
            <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-text-white mb-6">
              The Webotixs <span className="gradient-text">Journal</span>
            </h1>
            <p className="text-text-gray text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
              Guides, deep dives, and expert tutorials on modern design principles, Next.js scaling strategies, and business growth.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Blogs List */}
      <section className="py-16 md:py-24 border-t border-border/40 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {posts.map((post, i) => (
              <ScrollReveal key={post.id} delay={i * 0.1}>
                <Link href={`/blog/${post.slug}`} className="group block h-full">
                  <div className="h-full bg-background-card border border-border rounded-3xl overflow-hidden card-hover flex flex-col justify-between">
                    <div>
                      {/* Image placeholder or real image */}
                      <div className="relative h-48 bg-gradient-to-br from-background-section to-background-secondary overflow-hidden flex items-center justify-center border-b border-border/50">
                        {post.featured_image ? (
                          <img src={post.featured_image} alt={post.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                        ) : (
                          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from/25 to-primary-to/25 flex items-center justify-center border border-primary/20">
                            <span className="font-display font-bold text-2xl gradient-text">{post.title?.[0] || 'B'}</span>
                          </div>
                        )}
                        <div className="absolute top-4 left-4 z-10">
                          <span className="px-2.5 py-1 glass rounded-lg text-[10px] text-text-white border border-border/60 uppercase font-semibold">
                            {post.category?.name || 'Article'}
                          </span>
                        </div>
                      </div>

                      {/* Content */}
                      <div className="p-6">
                        {/* Meta */}
                        <div className="flex items-center gap-4 text-text-gray text-xs mb-3">
                          <span className="flex items-center gap-1">
                            <Calendar size={12} />
                            {post.published_at ? formatDate(post.published_at) : 'Draft'}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock size={12} />
                            {post.reading_time} min read
                          </span>
                        </div>

                        <h2 className="font-display text-xl font-bold text-text-white mb-2.5 group-hover:gradient-text transition-colors duration-300">
                          {post.title}
                        </h2>
                        <p className="text-text-gray text-sm leading-relaxed mb-4 line-clamp-3">
                          {post.excerpt}
                        </p>
                      </div>
                    </div>

                    <div className="px-6 pb-6">
                      <div className="flex items-center gap-2 text-primary text-xs font-semibold group-hover:underline">
                        Read Article
                        <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </div>
                </Link>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
