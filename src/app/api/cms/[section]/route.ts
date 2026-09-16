import { NextResponse } from 'next/server'
import { revalidatePath } from 'next/cache'
import { createAdminClient } from '@/lib/supabase/admin'
import {
  mockBlogPosts,
  mockPortfolio,
  mockServices,
  mockTestimonials,
  mockTeam,
  mockIndustries,
} from '@/lib/data/mock'

// Global server-side cache so changes persist immediately across all pages/devices
const globalStore: Record<string, any> = globalThis as any
if (!globalStore.__webotixs_cms_store) {
  globalStore.__webotixs_cms_store = {
    blogs: [...mockBlogPosts],
    portfolio: [...mockPortfolio],
    services: [...mockServices],
    testimonials: [...mockTestimonials],
    team: [...mockTeam],
    industries: [...mockIndustries],
    media: [],
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

export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET(
  request: Request,
  { params }: { params: Promise<{ section: string }> }
) {
  const { section } = await params

  try {
    const supabase = createAdminClient()
    const { data: fileData } = await supabase.storage.from('webotixs_cms_data').download(`${section}.json`)
    if (fileData) {
      const text = await fileData.text()
      const json = JSON.parse(text)
      if (json && (Array.isArray(json) ? json.length > 0 : Object.keys(json).length > 0)) {
        store[section] = json
        return NextResponse.json({ success: true, data: json, source: 'supabase_storage' })
      }
    }
  } catch (e) {
    // Supabase offline -> fallback cleanly to in-memory store
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

  // CRITICAL FIX: Download latest existing section state from Supabase Storage first
  // to ensure cold-started serverless instances never wipe existing images or user edits!
  try {
    const supabase = createAdminClient()
    const { data: fileData } = await supabase.storage.from('webotixs_cms_data').download(`${section}.json`)
    if (fileData) {
      const text = await fileData.text()
      const json = JSON.parse(text)
      if (json && (Array.isArray(json) ? json.length > 0 : Object.keys(json).length > 0)) {
        store[section] = json
      }
    }
  } catch (e) {}

  // Update in-memory server cache safely
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

  // Save merged state to Supabase Storage bucket for permanent cloud persistence
  try {
    const supabase = createAdminClient()
    const buffer = Buffer.from(JSON.stringify(store[section], null, 2))
    await supabase.storage.from('webotixs_cms_data').upload(`${section}.json`, buffer, {
      contentType: 'application/json',
      upsert: true,
    })
  } catch (e) {
    console.error('[CMS Save Error]:', e)
  }

  // Immediately revalidate Next.js server cache across all public and admin pages
  try {
    revalidatePath('/')
    revalidatePath('/blog')
    revalidatePath('/portfolio')
    revalidatePath('/services')
    revalidatePath('/industries')
    revalidatePath('/team')
    revalidatePath('/admin/services')
    revalidatePath('/admin/blogs')
    revalidatePath('/admin/portfolio')
    revalidatePath('/admin/media')
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

  // Fetch latest state first before deleting
  try {
    const supabase = createAdminClient()
    const { data: fileData } = await supabase.storage.from('webotixs_cms_data').download(`${section}.json`)
    if (fileData) {
      const text = await fileData.text()
      const json = JSON.parse(text)
      if (json && Array.isArray(json)) {
        store[section] = json
      }
    }
  } catch (e) {}

  store[section] = store[section].filter((item: any) => item.id !== id)

  // Update in Supabase Storage
  try {
    const supabase = createAdminClient()
    const buffer = Buffer.from(JSON.stringify(store[section], null, 2))
    await supabase.storage.from('webotixs_cms_data').upload(`${section}.json`, buffer, {
      contentType: 'application/json',
      upsert: true,
    })
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
