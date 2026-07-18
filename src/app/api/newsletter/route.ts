import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'
import { newsletterSchema } from '@/lib/validations/contact'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // Validate with Zod
    const parsed = newsletterSchema.safeParse(body)
    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid email address', details: parsed.error.flatten().fieldErrors },
        { status: 400 }
      )
    }

    const { email, name } = parsed.data

    const supabase = createAdminClient()

    // Check if subscriber already exists
    const { data: existing } = await supabase
      .from('newsletter_subscribers')
      .select('id, status')
      .eq('email', email)
      .single()

    if (existing) {
      if (existing.status === 'unsubscribed') {
        // Reactivate subscriber
        await supabase
          .from('newsletter_subscribers')
          .update({ status: 'active', name: name || undefined })
          .eq('id', existing.id)

        return NextResponse.json(
          { success: true, message: 'Welcome back! Your subscription has been reactivated.' },
          { status: 200 }
        )
      }

      return NextResponse.json(
        { success: true, message: 'You are already subscribed to our newsletter!' },
        { status: 200 }
      )
    }

    // Insert new subscriber
    const { error } = await supabase
      .from('newsletter_subscribers')
      .insert({
        email,
        name: name || null,
        status: 'active',
        source: 'website_footer',
      })

    if (error) {
      console.error('[Newsletter Supabase Error]:', error)
      return NextResponse.json(
        { error: 'Failed to subscribe. Please try again later.' },
        { status: 500 }
      )
    }

    // Optional: Send welcome email via Resend
    try {
      const resendKey = process.env.RESEND_API_KEY
      if (resendKey) {
        await fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${resendKey}`,
          },
          body: JSON.stringify({
            from: 'Webotixs <newsletter@webotixs.com>',
            to: [email],
            subject: 'Welcome to the Webotixs Innovation Dispatch 🚀',
            html: `
              <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; color: #1e293b;">
                <h1 style="color: #2563eb;">Welcome to Webotixs, ${name || 'Visionary'}!</h1>
                <p>Thank you for subscribing to our newsletter. We share cutting-edge insights on Next.js, AI workflows, enterprise web design, and digital product strategy.</p>
                <p>Stay tuned for our next edition!</p>
                <p style="margin-top: 32px; font-size: 12px; color: #64748b;">© 2026 Webotixs Agency. All rights reserved.</p>
              </div>
            `,
          }),
        })
      }
    } catch (emailError) {
      console.error('[Newsletter Welcome Email Error]:', emailError)
    }

    return NextResponse.json(
      { success: true, message: 'Thank you for subscribing! Check your inbox soon.' },
      { status: 201 }
    )
  } catch (e) {
    console.error('[Newsletter API Error]:', e)
    return NextResponse.json(
      { error: 'Internal server error.' },
      { status: 500 }
    )
  }
}
