import {
  mockBlogPosts,
  mockPortfolio,
  mockServices,
  mockTestimonials,
  mockTeam,
  mockIndustries,
} from '@/lib/data/mock'

const defaultHomepage = {
  hero: {
    badge: 'Trusted by 200+ Global Clients',
    titlePrefix: 'We Build',
    titleHighlight: 'Digital',
    titleSuffix: 'Experiences',
    subtitle:
      'We build digital experiences that drive measurable business growth — combining bespoke web design, custom mobile engineering, and brand strategy. We partner with ambitious leaders to turn their vision into high-performance digital products and unforgettable user experiences.',
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
}

const fallbackMap: Record<string, any> = {
  blogs: mockBlogPosts,
  portfolio: mockPortfolio,
  services: mockServices,
  testimonials: mockTestimonials,
  team: mockTeam,
  industries: mockIndustries,
  homepage: defaultHomepage,
}

function safeSetLocalStorage(key: string, data: any) {
  try {
    localStorage.setItem(key, JSON.stringify(data))
  } catch (e) {
    // If quota exceeded, try cleaning legacy base64 strings and retry
    try {
      const cleanData = JSON.parse(JSON.stringify(data), (k, v) => {
        if (typeof v === 'string' && v.startsWith('data:image')) return ''
        return v
      })
      localStorage.setItem(key, JSON.stringify(cleanData))
    } catch {}
  }
}

export async function getCMSData<T = any>(section: string): Promise<T> {
  if (typeof window !== 'undefined') {
    try {
      const res = await fetch(`/api/cms/${section}`, { cache: 'no-store' })
      if (res.ok) {
        const json = await res.json()
        if (json.success && json.data && (Array.isArray(json.data) ? json.data.length > 0 : Object.keys(json.data).length > 0)) {
          safeSetLocalStorage(`webotixs_cms_${section}`, json.data)
          return json.data as T
        }
      }
    } catch {}

    try {
      const local = localStorage.getItem(`webotixs_cms_${section}`)
      if (local) {
        const parsed = JSON.parse(local)
        if (parsed && (Array.isArray(parsed) ? parsed.length > 0 : Object.keys(parsed).length > 0)) {
          return parsed as T
        }
      }
    } catch {}
  } else {
    // Server-side environment: query Supabase Cloud Storage bucket directly for universal cross-domain sync
    try {
      const globalStore: any = globalThis as any
      const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
      const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

      if (supabaseUrl && supabaseKey) {
        const { createClient } = await import('@supabase/supabase-js')
        const supabase = createClient(supabaseUrl, supabaseKey, {
          auth: { autoRefreshToken: false, persistSession: false },
        })

        const { data: fileData } = await supabase.storage.from('webotixs_cms_data').download(`${section}.json`)
        if (fileData) {
          const text = await fileData.text()
          const json = JSON.parse(text)
          if (json && (Array.isArray(json) ? json.length > 0 : Object.keys(json).length > 0)) {
            if (!globalStore.__webotixs_cms_store) globalStore.__webotixs_cms_store = {}
            globalStore.__webotixs_cms_store[section] = json
            return json as T
          }
        }
      }
    } catch {}
  }

  return (fallbackMap[section] || []) as T
}

export async function saveCMSItem(section: string, item: any): Promise<any> {
  if (typeof window !== 'undefined') {
    try {
      const local = localStorage.getItem(`webotixs_cms_${section}`)
      let current = local ? JSON.parse(local) : [...(fallbackMap[section] || [])]
      if (Array.isArray(current)) {
        const idx = current.findIndex((i: any) => i.id === item.id)
        if (idx >= 0) current[idx] = item
        else current.push(item)
        safeSetLocalStorage(`webotixs_cms_${section}`, current)
        window.dispatchEvent(new Event('storage'))
      }
    } catch {}
  }

  try {
    const res = await fetch(`/api/cms/${section}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item }),
    })
    const json = await res.json()
    if (json.success && typeof window !== 'undefined') {
      safeSetLocalStorage(`webotixs_cms_${section}`, json.data)
      window.dispatchEvent(new Event('storage'))
      return json.data
    }
  } catch {}

  return null
}

export async function saveCMSList(section: string, data: any): Promise<any> {
  if (typeof window !== 'undefined') {
    safeSetLocalStorage(`webotixs_cms_${section}`, data)
    window.dispatchEvent(new Event('storage'))
  }

  try {
    const res = await fetch(`/api/cms/${section}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data }),
    })
    const json = await res.json()
    if (json.success && typeof window !== 'undefined') {
      safeSetLocalStorage(`webotixs_cms_${section}`, json.data)
      window.dispatchEvent(new Event('storage'))
      return json.data
    }
  } catch {}

  return data
}

export async function deleteCMSItem(section: string, id: string): Promise<any> {
  if (typeof window !== 'undefined') {
    try {
      const local = localStorage.getItem(`webotixs_cms_${section}`)
      if (local && Array.isArray(JSON.parse(local))) {
        const filtered = JSON.parse(local).filter((i: any) => i.id !== id)
        safeSetLocalStorage(`webotixs_cms_${section}`, filtered)
        window.dispatchEvent(new Event('storage'))
      }
    } catch {}
  }

  try {
    const res = await fetch(`/api/cms/${section}?id=${id}`, { method: 'DELETE' })
    const json = await res.json()
    if (json.success && typeof window !== 'undefined') {
      safeSetLocalStorage(`webotixs_cms_${section}`, json.data)
      window.dispatchEvent(new Event('storage'))
      return json.data
    }
  } catch {}

  return null
}
