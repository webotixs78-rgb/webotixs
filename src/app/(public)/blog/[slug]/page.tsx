import { notFound } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, Calendar, Clock, User, Bookmark } from 'lucide-react'
import { getCMSData } from '@/lib/data/cms'
import { formatDate } from '@/lib/utils'
import ScrollReveal from '@/components/animations/ScrollReveal'

export const dynamic = 'force-dynamic'

interface Props {
  params: Promise<{ slug: string }>
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params
  const posts: any[] = await getCMSData('blogs')
  const post = posts.find((p: any) => p.slug === slug)

  if (!post) {
    notFound()
  }

  return (
    <div className="pt-24 bg-background min-h-screen">
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Back Link */}
        <Link
          href="/blog"
          className="inline-flex items-center gap-2 text-text-gray hover:text-text-white mb-8 text-sm group transition-colors"
        >
          <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
          Back to Journal
        </Link>

        {/* Post Meta */}
        <ScrollReveal>
          <div className="flex flex-wrap items-center gap-4 text-text-gray text-sm mb-6">
            <span className="px-3 py-1 glass rounded-full text-xs text-primary border border-primary/25 uppercase font-semibold">
              {post.category?.name || 'Article'}
            </span>
            <span className="flex items-center gap-1.5">
              <Calendar size={14} />
              {post.published_at ? formatDate(post.published_at) : 'Draft'}
            </span>
            <span className="flex items-center gap-1.5">
              <Clock size={14} />
              {post.reading_time || 5} min read
            </span>
          </div>

          <h1 className="font-display text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-text-white mb-8 leading-tight">
            {post.title}
          </h1>

          {/* Author Badge */}
          <div className="flex items-center gap-3.5 border-b border-border/40 pb-8 mb-12">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary-from/25 to-primary-to/25 flex items-center justify-center border border-primary/20">
              <User size={18} className="text-primary" />
            </div>
            <div>
              <div className="font-display font-semibold text-text-white text-sm">Webotixs Team</div>
              <div className="text-text-gray text-xs">Technical Authors</div>
            </div>
          </div>
        </ScrollReveal>

        {/* Content area */}
        <ScrollReveal className="prose prose-invert max-w-none text-text-gray leading-relaxed text-base space-y-6 tiptap-content">
          <p className="text-lg text-text-white font-medium mb-8 leading-relaxed">
            {post.excerpt}
          </p>
          <div dangerouslySetInnerHTML={{ __html: post.content }} />
        </ScrollReveal>

        {/* Tag list */}
        <ScrollReveal className="mt-12 pt-8 border-t border-border/40">
          <div className="flex flex-wrap items-center gap-2">
            <Bookmark size={14} className="text-text-gray mr-1" />
            {(post.tags || []).map((tag: string, idx: number) => (
              <span key={idx} className="px-3 py-1 bg-background-card border border-border/60 rounded-xl text-text-gray text-xs">
                #{tag}
              </span>
            ))}
          </div>
        </ScrollReveal>
      </article>
    </div>
  )
}
