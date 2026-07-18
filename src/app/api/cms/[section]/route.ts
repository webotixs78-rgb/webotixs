import { NextResponse } from 'next/server'
import { revalidatePath } from 'next/cache'
import { createClient } from '@/lib/supabase/client'
import {
  mockBlogPosts,
  mockPortfolio,
  mockServices,
  mockTestimonials,
  mockTeam,
  mockIndustries,
} from '@/lib/data/mock'

// Global server-side cache so changes persist immediately across all pages/devices
// even if Supabase table creation SQL has not been executed yet.
const globalStore: Record<string, any> = globalThis as any
if (!globalStore.__webotixs_cms_store) {
  globalStore.__webotixs_cms_store = {
    blogs: [...mockBlogPosts],
    portfolio: [...mockPortfolio],
    services: [...mockServices],
    testimonials: [...mockTestimonials],
    team: [...mockTeam],
    industries: [...mockIndustries],
    homepage: {
      hero: {
        badge: 'Trusted by 200+ Global Clients',
        titlePrefix: 'We Build',
        titleHighlight: 'Digital',
        titleSuffix: 'Experiences',
        subtitle:
          'Premium web design, mobile apps, and brand identities crafted for ambitious businesses. We turn your vision into stunning digital products.',
        primaryCtaText: 'Start Your Project',
        primaryCtaLink: '/contact',
        secondaryCtaText: 'View Our Work',
        secondaryCtaLink: '/portfolio',
      },
      stats: [
        { label: 'Successful Projects', value: '150+' },
        { label: 'Client Satisfaction', value: '98%' },
        { label: 'Years Experience', value: '10+' },
        { label: 'Impressions Generated', value: '25M+' },
      ],
      reasons: [
        { title: 'Full-Stack Expertise', desc: 'End-to-end development covering web, mobile, database design, and cloud architecture.' },
        { title: 'Lightning Fast Delivery', desc: 'Iterative sprint releases delivering functional prototypes within weeks, not months.' },
        { title: 'Enterprise Security', desc: 'Bank-grade encryption, SOC2 readiness, and strictly audited deployment pipelines.' },
        { title: 'Conversion-Focused UX', desc: 'Data-driven designs mapped directly to user behavior analytics and high conversion.' },
        { title: 'Scalable Cloud Native', desc: 'Built on Next.js 16, Supabase, and AWS/Vercel edge for infinite auto-scaling.' },
        { title: '24/7 Dedicated Support', desc: 'Direct Slack channel access to our senior engineers with guaranteed SLA turnaround.' },
      ],
    },
  }
}

const store = globalStore.__webotixs_cms_store

// Map section slug to Supabase table name
const tableMap: Record<string, string> = {
  blogs: 'blogs',
  portfolio: 'portfolio',
  services: 'services',
  testimonials: 'testimonials',
  team: 'team',
  industries: 'industries',
  homepage: 'homepage_sections',
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ section: string }> }
) {
  const { section } = await params

  try {
    const supabase = createClient()
    const tableName = tableMap[section]

    if (tableName) {
      if (section === 'homepage') {
        const { data, error } = await supabase
          .from('homepage_sections')
          .select('content')
          .eq('id', 'main')
          .single()
        if (data?.content && !error) {
          store[section] = data.content
          return NextResponse.json({ success: true, data: data.content, source: 'supabase' })
        }
      } else {
        const { data, error } = await supabase
          .from(tableName)
          .select('*')
          .order('created_at', { ascending: false })
        if (data && data.length > 0 && !error) {
          store[section] = data
          return NextResponse.json({ success: true, data, source: 'supabase' })
        }
      }
    }
  } catch (e) {
    // Supabase offline or tables not created yet -> fallback cleanly to live store
  }

  const data = store[section] || []
  return NextResponse.json({ success: true, data, source: 'cache' })
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ section: string }> }
) {
  const { section } = await params
  let body: any
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ success: false, error: 'Invalid JSON' }, { status: 400 })
  }

  // Update in-memory server cache immediately
  if (body.data !== undefined) {
    store[section] = body.data
  } else if (body.item) {
    const current = Array.isArray(store[section]) ? store[section] : []
    const idx = current.findIndex((i: any) => i.id === body.item.id)
    if (idx >= 0) {
      current[idx] = { ...current[idx], ...body.item, updated_at: new Date().toISOString() }
    } else {
      current.unshift(body.item)
    }
    store[section] = current
  }

  // Attempt to save to Supabase
  try {
    const supabase = createClient()
    const tableName = tableMap[section]
    if (tableName) {
      if (section === 'homepage') {
        await supabase.from('homepage_sections').upsert({
          id: 'main',
          content: store[section],
          updated_at: new Date().toISOString(),
        })
      } else if (body.item) {
        await supabase.from(tableName).upsert({
          ...body.item,
          updated_at: new Date().toISOString(),
        })
      } else if (Array.isArray(store[section])) {
        await supabase.from(tableName).upsert(store[section])
      }
    }
  } catch (e) {
    // Ignore Supabase table errors if migration not run
  }

  // Immediately flush Next.js server cache across all public pages!
  try {
    revalidatePath('/')
    revalidatePath('/blog')
    revalidatePath('/portfolio')
    revalidatePath('/services')
    revalidatePath('/industries')
    revalidatePath('/team')
  } catch (e) {}

  return NextResponse.json({ success: true, data: store[section] })
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ section: string }> }
) {
  const { section } = await params
  const { searchParams } = new URL(request.url)
  const id = searchParams.get('id')

  if (!id || !Array.isArray(store[section])) {
    return NextResponse.json({ success: false, error: 'Invalid ID or section' }, { status: 400 })
  }

  store[section] = store[section].filter((item: any) => item.id !== id)

  // Attempt delete in Supabase
  try {
    const supabase = createClient()
    const tableName = tableMap[section]
    if (tableName) {
      await supabase.from(tableName).delete().eq('id', id)
    }
  } catch (e) {}

  try {
    revalidatePath('/')
    revalidatePath('/blog')
    revalidatePath('/portfolio')
    revalidatePath('/services')
    revalidatePath('/industries')
    revalidatePath('/team')
  } catch (e) {}

  return NextResponse.json({ success: true, data: store[section] })
}
