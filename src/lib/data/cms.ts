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

export async function getCMSData<T = any>(section: string): Promise<T> {
  // If running on browser client, check localStorage first for instant display, then fetch API
  if (typeof window !== 'undefined') {
    try {
      const local = localStorage.getItem(`webotixs_cms_${section}`)
      if (local) {
        const parsed = JSON.parse(local)
        if (parsed && (Array.isArray(parsed) ? parsed.length > 0 : Object.keys(parsed).length > 0)) {
          // Trigger background fetch to keep sync
          fetch(`/api/cms/${section}`)
            .then((r) => r.json())
            .then((res) => {
              if (res?.success && res.data) {
                localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(res.data))
              }
            })
            .catch(() => {})
          return parsed
        }
      }
    } catch {}

    try {
      const res = await fetch(`/api/cms/${section}`, { cache: 'no-store' })
      if (res.ok) {
        const json = await res.json()
        if (json.success && json.data) {
          localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(json.data))
          return json.data
        }
      }
    } catch {}
  } else {
    // Server-side environment
    try {
      const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'
      const res = await fetch(`${baseUrl}/api/cms/${section}`, { cache: 'no-store' })
      if (res.ok) {
        const json = await res.json()
        if (json.success && json.data) {
          return json.data
        }
      }
    } catch {}
  }

  return fallbackMap[section] || []
}

export async function saveCMSItem(section: string, item: any): Promise<any> {
  if (typeof window !== 'undefined') {
    // Immediately update localStorage
    try {
      const local = localStorage.getItem(`webotixs_cms_${section}`)
      let current = local ? JSON.parse(local) : [...(fallbackMap[section] || [])]
      if (Array.isArray(current)) {
        const idx = current.findIndex((i: any) => i.id === item.id)
        if (idx >= 0) current[idx] = item
        else current.unshift(item)
        localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(current))
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
      localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(json.data))
      return json.data
    }
  } catch {}

  return null
}

export async function saveCMSList(section: string, data: any): Promise<any> {
  if (typeof window !== 'undefined') {
    localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(data))
  }

  try {
    const res = await fetch(`/api/cms/${section}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data }),
    })
    const json = await res.json()
    if (json.success && typeof window !== 'undefined') {
      localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(json.data))
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
        localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(filtered))
      }
    } catch {}
  }

  try {
    const res = await fetch(`/api/cms/${section}?id=${id}`, { method: 'DELETE' })
    const json = await res.json()
    if (json.success && typeof window !== 'undefined') {
      localStorage.setItem(`webotixs_cms_${section}`, JSON.stringify(json.data))
      return json.data
    }
  } catch {}

  return null
}
