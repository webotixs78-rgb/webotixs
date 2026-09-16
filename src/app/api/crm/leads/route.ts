import { NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function GET() {
  try {
    const supabase = createAdminClient()
    const { data, error } = await supabase
      .from('contact_inquiries')
      .select('*')
      .order('created_at', { ascending: false })

    if (!error && data) {
      const formattedLeads = data.map((item: any) => ({
        id: item.id,
        name: item.name,
        email: item.email,
        phone: item.phone || 'N/A',
        company: item.company || 'N/A',
        service: item.service || 'Web Design & Development',
        message: item.message,
        status: item.status === 'New' ? 'New Lead' : item.status || 'New Lead',
        priority: 'Medium',
        source: item.inquiry_source || 'Website',
        assigned_to: 'Admin',
        created_at: item.created_at,
        ip_address: item.ip_address,
        browser: item.browser,
        country: item.country,
      }))
      return NextResponse.json({ leads: formattedLeads })
    }

    // Fallback to memory / submissions
    const { data: subData } = await supabase
      .from('contact_submissions')
      .select('*')
      .order('created_at', { ascending: false })

    if (subData && subData.length > 0) {
      const formatted = subData.map((s: any) => ({
        id: s.id,
        name: s.name,
        email: s.email,
        phone: s.phone || 'N/A',
        company: s.company || 'N/A',
        service: s.service || 'General',
        message: s.message,
        status: 'New Lead',
        priority: s.ai_priority ? (s.ai_priority === 'high' ? 'High' : 'Medium') : 'Medium',
        source: 'Website',
        assigned_to: 'Admin',
        created_at: s.created_at,
      }))
      return NextResponse.json({ leads: formatted })
    }

    return NextResponse.json({ leads: [] })
  } catch (err) {
    return NextResponse.json({ leads: [] })
  }
}
