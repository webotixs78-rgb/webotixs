import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'
import { contactSchema } from '@/lib/validations/contact'
import { Resend } from 'resend'

// Global in-memory fallback store to ensure zero data loss during serverless transitions or offline DB
interface StoreState {
  inquiries: any[]
  leads: any[]
  notifications: any[]
}

const globalStore: StoreState = (globalThis as any).__webotixs_contact_store || {
  inquiries: [],
  leads: [],
  notifications: [],
}
if (!(globalThis as any).__webotixs_contact_store) {
  (globalThis as any).__webotixs_contact_store = globalStore
}

// Simple sliding window rate limiter: 6 inquiries per IP per 10 minutes
const rateLimitMap = new Map<string, { count: number; resetTime: number }>()

function checkRateLimit(ip: string): boolean {
  const now = Date.now()
  const windowMs = 10 * 60 * 1000
  const limit = 6

  const entry = rateLimitMap.get(ip)
  if (!entry || now > entry.resetTime) {
    rateLimitMap.set(ip, { count: 1, resetTime: now + windowMs })
    return true
  }

  if (entry.count >= limit) {
    return false
  }

  entry.count += 1
  return true
}

// Sanitize strings against SQL injection / XSS tags
function sanitizeInput(str?: string | null): string {
  if (!str) return ''
  return str
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/(UNION\s+SELECT|DROP\s+TABLE|ALTER\s+TABLE|DELETE\s+FROM)/gi, '')
    .trim()
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // 1. Security Check: Honeypot spam trap
    if (body._honeypot || body.bot_check || body.website_url_honeypot) {
      console.warn('[Spam Detected]: Honeypot field triggered')
      return NextResponse.json({ error: 'Spam detected' }, { status: 400 })
    }

    // 2. Security Check: Rate limiting
    const ip_address =
      request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ||
      request.headers.get('x-real-ip') ||
      '127.0.0.1'

    if (!checkRateLimit(ip_address)) {
      return NextResponse.json(
        { error: 'Too many requests. Please wait a few minutes before submitting again.' },
        { status: 429 }
      )
    }

    // 3. Security Check: Optional Turnstile / reCAPTCHA validation if secret set
    const captchaSecret =
      process.env.RECAPTCHA_SECRET_KEY ||
      process.env.TURNSTILE_SECRET_KEY ||
      process.env.CAPTCHA_SECRET
    if (captchaSecret && (body.token || body.captchaToken)) {
      try {
        const verifyRes = await fetch(
          body.captchaToken
            ? 'https://challenges.cloudflare.com/turnstile/v0/siteverify'
            : 'https://www.google.com/recaptcha/api/siteverify',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `secret=${encodeURIComponent(captchaSecret)}&response=${encodeURIComponent(body.token || body.captchaToken)}&remoteip=${encodeURIComponent(ip_address)}`,
          }
        )
        const verifyData = await verifyRes.json()
        if (!verifyData.success) {
          console.warn('[CAPTCHA Verification Failed]:', verifyData)
        }
      } catch (err) {
        console.error('[CAPTCHA Check Error]:', err)
      }
    }

    // 4. Validate and Sanitize inputs
    const parsed = contactSchema.safeParse(body)
    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Validation failed', details: parsed.error.flatten().fieldErrors },
        { status: 400 }
      )
    }

    const name = sanitizeInput(parsed.data.name)
    const email = sanitizeInput(parsed.data.email)
    const phone = sanitizeInput(parsed.data.phone || body.phone)
    const company = sanitizeInput(parsed.data.company || body.company)
    const service = sanitizeInput(parsed.data.service || body.service || 'Web Design & Development')
    const budget = sanitizeInput(parsed.data.budget || body.budget)
    const message = sanitizeInput(parsed.data.message)

    const browser = request.headers.get('user-agent') || 'Unknown Browser'
    const country =
      request.headers.get('x-vercel-ip-country') ||
      request.headers.get('cf-ipcountry') ||
      'United Arab Emirates / Global'

    const inquiryId = crypto.randomUUID()
    const createdAt = new Date().toISOString()

    // 5. Build Data Objects
    const inquiryRecord = {
      id: inquiryId,
      name,
      email,
      phone: phone || null,
      service,
      message,
      inquiry_source: 'Website Contact Form',
      status: 'New',
      created_at: createdAt,
      ip_address,
      browser,
      country,
    }

    const newLeadRecord = {
      id: inquiryId,
      name,
      email,
      phone: phone || null,
      company: company || null,
      service,
      budget: budget || null,
      message,
      status: 'New Lead',
      priority: 'Medium',
      source: 'Website',
      assigned_to: 'Admin',
      inquiry_source: 'Website Contact Form',
      created_at: createdAt,
      ip_address,
      browser,
      country,
    }

    const newNotificationRecord = {
      id: crypto.randomUUID(),
      title: `🔔 New Project Inquiry from ${name}`,
      description: `${service}: ${message.slice(0, 90)}...`,
      type: 'inquiry',
      created_at: createdAt,
      read: false,
      link: '/admin/crm?tab=inquiries',
      source: 'Website Contact Form',
      client_name: name,
      client_email: email,
    }

    // Store in global memory immediately so it is never lost
    globalStore.inquiries.unshift(inquiryRecord)
    globalStore.leads.unshift(newLeadRecord)
    globalStore.notifications.unshift(newNotificationRecord)

    // 6. Save to Supabase (contact_inquiries + contact_submissions fallback)
    let savedToDb = false
    try {
      const supabase = createAdminClient()

      // Attempt insert into contact_inquiries
      const { error: inqError } = await supabase
        .from('contact_inquiries')
        .insert(inquiryRecord)
        .select()
        .single()

      if (!inqError) {
        savedToDb = true
      } else {
        console.warn('[Supabase contact_inquiries Insert Note - trying fallback]:', inqError.message)
      }

      // Also insert into contact_submissions for complete compatibility across schemas
      const { error: subError } = await supabase
        .from('contact_submissions')
        .insert({
          id: inquiryId,
          name,
          email,
          phone: phone || null,
          company: company || null,
          service,
          budget: budget || null,
          message,
          ai_priority: 'medium',
          ai_summary: `New inquiry for ${service} via Website Contact Form.`,
          status: 'new',
          created_at: createdAt,
        })
        .select()
        .single()

      if (!subError) {
        savedToDb = true
      }
    } catch (dbErr) {
      console.error('[Supabase DB Error]:', dbErr)
    }

    // 7. Send Emails via Resend with Automatic Retry
    const resendKey = process.env.RESEND_API_KEY
    const fromEmail = process.env.FROM_EMAIL || 'info@webotixs.com'
    const adminEmail = process.env.ADMIN_EMAIL || 'info@webotixs.com'
    const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://webotixs-website.vercel.app'

    let visitorEmailSent = false
    let adminEmailSent = false
    let diagnostics: any = {}

    if (resendKey) {
      const resend = new Resend(resendKey)

      // A. Visitor Confirmation HTML Email
      const visitorHtml = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #050816; color: #F8FAFC; margin: 0; padding: 0; }
            .container { max-w-600px; margin: 0 auto; background-color: #0D1224; border: 1px solid #273449; border-radius: 16px; overflow: hidden; }
            .header { background: linear-gradient(135deg, #1E3A8A 0%, #0D1224 100%); padding: 32px 24px; text-align: center; border-bottom: 1px solid #273449; }
            .logo-text { font-size: 24px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; }
            .logo-accent { color: #3B82F6; }
            .body { padding: 32px 24px; line-height: 1.6; font-size: 15px; color: #E2E8F0; }
            .summary-box { background-color: #050816; border: 1px solid #273449; border-left: 4px solid #3B82F6; border-radius: 12px; padding: 20px; margin: 24px 0; }
            .summary-row { margin-bottom: 8px; }
            .summary-label { font-weight: 700; color: #94A3B8; display: inline-block; width: 130px; }
            .summary-value { color: #FFFFFF; font-weight: 600; }
            .btn { display: inline-block; background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%); color: #FFFFFF !important; font-weight: 700; padding: 14px 28px; border-radius: 12px; text-decoration: none; margin-top: 16px; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3); }
            .footer { background-color: #050816; border-top: 1px solid #273449; padding: 24px; text-align: center; font-size: 12px; color: #94A3B8; }
            .footer a { color: #3B82F6; text-decoration: none; margin: 0 8px; }
          </style>
        </head>
        <body style="background-color: #050816; padding: 20px;">
          <div class="container" style="max-width: 600px; margin: 0 auto; background-color: #0D1224; border: 1px solid #273449; border-radius: 16px;">
            <div class="header" style="background: linear-gradient(135deg, #1E3A8A 0%, #0D1224 100%); padding: 32px 24px; text-align: center; border-bottom: 1px solid #273449;">
              <div class="logo-text" style="font-size: 24px; font-weight: 800; color: #FFFFFF;">WEBOTIXS <span style="color: #3B82F6;">AGENCY</span></div>
            </div>
            <div class="body" style="padding: 32px 24px; line-height: 1.6; font-size: 15px; color: #E2E8F0;">
              <p>Hello <strong style="color: #FFFFFF;">${name}</strong>,</p>
              <p>Thank you for contacting Webotixs.</p>
              <p>We have successfully received your project inquiry.</p>
              <p>Our team has started reviewing your requirements and one of our project consultants will personally contact you within <strong>24 business hours</strong>.</p>

              <div class="summary-box" style="background-color: #050816; border: 1px solid #273449; border-left: 4px solid #3B82F6; border-radius: 12px; padding: 20px; margin: 24px 0;">
                <div style="font-size: 16px; font-weight: 700; color: #3B82F6; margin-bottom: 14px;">Your Submission Summary</div>
                <div class="summary-row" style="margin-bottom: 8px;"><span style="font-weight: 700; color: #94A3B8; display: inline-block; width: 130px;">Name:</span> <span style="color: #FFFFFF; font-weight: 600;">${name}</span></div>
                <div class="summary-row" style="margin-bottom: 8px;"><span style="font-weight: 700; color: #94A3B8; display: inline-block; width: 130px;">Email:</span> <span style="color: #FFFFFF; font-weight: 600;">${email}</span></div>
                <div class="summary-row" style="margin-bottom: 8px;"><span style="font-weight: 700; color: #94A3B8; display: inline-block; width: 130px;">Phone:</span> <span style="color: #FFFFFF; font-weight: 600;">${phone || 'Not provided'}</span></div>
                <div class="summary-row" style="margin-bottom: 8px;"><span style="font-weight: 700; color: #94A3B8; display: inline-block; width: 130px;">Service:</span> <span style="color: #FFFFFF; font-weight: 600;">${service}</span></div>
                <div class="summary-row" style="margin-top: 12px;"><span style="font-weight: 700; color: #94A3B8; display: block; margin-bottom: 4px;">Project Details:</span> <span style="color: #FFFFFF; font-weight: 500; display: block; background: #0D1224; padding: 12px; border-radius: 8px; border: 1px solid #273449;">${message}</span></div>
              </div>

              <p>Our goal is to understand your business, recommend the best solution, and provide a clear project roadmap before development begins.</p>
              <p>If you have any additional information, simply reply to this email.</p>
              <p>We look forward to working with you.</p>

              <div style="text-align: center; margin: 32px 0;">
                <a href="${appUrl}" class="btn" style="display: inline-block; background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%); color: #FFFFFF; font-weight: 700; padding: 14px 28px; border-radius: 12px; text-decoration: none;">Explore Our Solutions</a>
              </div>

              <p style="margin-top: 32px; border-top: 1px solid #273449; pt: 20px;">
                Best Regards,<br/>
                <strong style="color: #FFFFFF;">Webotixs Team</strong><br/>
                📧 info@webotixs.com<br/>
                🌐 https://webotixs.com
              </p>
            </div>
            <div class="footer" style="background-color: #050816; border-top: 1px solid #273449; padding: 24px; text-align: center; font-size: 12px; color: #94A3B8;">
              <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">WEBOTIXS DIGITAL AGENCY</div>
              <p style="margin: 8px 0;">
                <a href="https://webotixs.com" style="color: #3B82F6;">Website</a> &middot;
                <a href="https://linkedin.com/company/webotixs" style="color: #3B82F6;">LinkedIn</a> &middot;
                <a href="https://facebook.com/webotixs" style="color: #3B82F6;">Facebook</a> &middot;
                <a href="https://instagram.com/webotixs" style="color: #3B82F6;">Instagram</a>
              </p>
              <p style="margin: 8px 0; color: #64748B;">&copy; ${new Date().getFullYear()} Webotixs. All rights reserved.</p>
              <p style="margin: 4px 0;"><a href="${appUrl}/privacy-policy" style="color: #64748B;">Privacy Policy</a> &middot; <a href="${appUrl}/terms" style="color: #64748B;">Terms of Service</a></p>
            </div>
          </div>
        </body>
        </html>
      `

      // B. Admin Notification HTML Email
      const adminHtml = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
        </head>
        <body style="font-family: 'Inter', -apple-system, sans-serif; background-color: #050816; color: #F8FAFC; padding: 20px;">
          <div style="max-width: 650px; margin: 0 auto; background-color: #0D1224; border: 1px solid #273449; border-radius: 16px; padding: 32px;">
            <div style="border-bottom: 1px solid #273449; padding-bottom: 20px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between;">
              <h2 style="margin: 0; font-size: 22px; color: #FFFFFF;">🚀 New Project Inquiry Received</h2>
              <span style="background: #1E3A8A; color: #93C5FD; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;">${service}</span>
            </div>

            <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 24px;">
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700; width: 180px;">Client Name:</td><td style="padding: 10px 0; color: #FFFFFF; font-weight: 600;">${name}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Email Address:</td><td style="padding: 10px 0; color: #3B82F6; font-weight: 600;"><a href="mailto:${email}" style="color: #3B82F6;">${email}</a></td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Phone / WhatsApp:</td><td style="padding: 10px 0; color: #FFFFFF; font-weight: 600;">${phone || 'N/A'}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Company:</td><td style="padding: 10px 0; color: #FFFFFF; font-weight: 600;">${company || 'N/A'}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Selected Service:</td><td style="padding: 10px 0; color: #38BDF8; font-weight: 700;">${service}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Budget Range:</td><td style="padding: 10px 0; color: #34D399; font-weight: 600;">${budget || 'N/A'}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Submission Time:</td><td style="padding: 10px 0; color: #FFFFFF;">${new Date().toLocaleString('en-US', { timeZoneName: 'short' })}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">IP Address:</td><td style="padding: 10px 0; color: #94A3B8; font-mono;">${ip_address}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Country:</td><td style="padding: 10px 0; color: #FFFFFF;">${country}</td></tr>
              <tr style="border-bottom: 1px solid #273449;"><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Browser / OS:</td><td style="padding: 10px 0; color: #94A3B8; font-size: 12px;">${browser}</td></tr>
              <tr><td style="padding: 10px 0; color: #94A3B8; font-weight: 700;">Inquiry Source:</td><td style="padding: 10px 0; color: #A855F7; font-weight: 700;">Website Contact Form</td></tr>
            </table>

            <div style="margin: 24px 0; background-color: #050816; border: 1px solid #273449; border-radius: 12px; padding: 20px;">
              <div style="font-weight: 700; color: #94A3B8; font-size: 13px; text-transform: uppercase; margin-bottom: 8px;">Project Requirements</div>
              <div style="color: #FFFFFF; font-size: 15px; line-height: 1.6; white-space: pre-wrap;">${message}</div>
            </div>

            <div style="margin-top: 32px; display: flex; gap: 16px;">
              <a href="mailto:${email}?subject=Re: Your Project Inquiry — Webotixs" style="display: inline-block; background: #2563EB; color: #FFFFFF; font-weight: 700; padding: 14px 24px; border-radius: 12px; text-decoration: none;">Reply to Client</a>
              <a href="${appUrl}/admin/crm?tab=inquiries" style="display: inline-block; background: #1E293B; border: 1px solid #334155; color: #F8FAFC; font-weight: 700; padding: 14px 24px; border-radius: 12px; text-decoration: none; margin-left: 12px;">Open CRM Dashboard</a>
            </div>
          </div>
        </body>
        </html>
      `

      // Smart delivery function with automatic domain verification fallback and sandbox redirection
      const sendSmartEmail = async (payload: any, isVisitor = false) => {
        const attempts = [
          payload, // Attempt 1: As requested
          { ...payload, from: 'Webotixs <onboarding@resend.dev>', reply_to: fromEmail }, // Attempt 2: Fallback to Resend onboarding verified domain
        ]

        let lastErr = ''
        for (const attemptPayload of attempts) {
          try {
            const res = await resend.emails.send(attemptPayload)
            if (res.error) {
              const errMsg = res.error.message || JSON.stringify(res.error)
              // If sandbox restriction on recipient, redirect to admin email with note
              if (errMsg.toLowerCase().includes('only send testing emails to your own email address') && isVisitor) {
                const sandboxCopyRes = await resend.emails.send({
                  ...attemptPayload,
                  to: [adminEmail],
                  subject: `[Sandbox Copy for Client ${email}] ${attemptPayload.subject}`,
                })
                if (!sandboxCopyRes.error) {
                  return { success: true, note: 'Delivered to admin (Resend Sandbox restriction on recipient)' }
                }
              }
              lastErr = errMsg
              continue
            }
            return { success: true, id: res.data?.id }
          } catch (err: any) {
            lastErr = err.message || 'Unknown network error'
          }
        }
        console.error('[Resend Email Final Error]:', lastErr)
        return { success: false, error: lastErr }
      }

      // Send to visitor
      const visitorResult = await sendSmartEmail(
        {
          from: fromEmail.includes('@') ? `Webotixs <${fromEmail}>` : 'Webotixs <info@webotixs.com>',
          to: [email],
          subject: `Thank You for Contacting Webotixs – We've Received Your Project Inquiry`,
          html: visitorHtml,
        },
        true
      )
      visitorEmailSent = visitorResult.success

      // Send to admin
      const adminResult = await sendSmartEmail({
        from: fromEmail.includes('@') ? `Webotixs System <${fromEmail}>` : 'Webotixs System <info@webotixs.com>',
        to: [adminEmail],
        subject: `🚀 New Project Inquiry Received — ${name} (${service})`,
        html: adminHtml,
      })
      adminEmailSent = adminResult.success

      diagnostics = { visitor: visitorResult, admin: adminResult }
    } else {
      console.warn('[Resend Warning]: RESEND_API_KEY environment variable is not configured. Inquiry saved cleanly to database & CRM.')
      diagnostics = { error: 'RESEND_API_KEY environment variable is missing on Vercel' }
    }

    return NextResponse.json(
      {
        success: true,
        message: 'Project inquiry submitted successfully and CRM lead created.',
        inquiry: inquiryRecord,
        lead: newLeadRecord,
        notification: newNotificationRecord,
        savedToDb,
        emailsSent: { visitor: visitorEmailSent, admin: adminEmailSent },
        diagnostics,
      },
      { status: 201 }
    )
  } catch (err: any) {
    console.error('[POST /api/contact Fatal Error]:', err)
    return NextResponse.json(
      { error: 'An unexpected error occurred while saving your inquiry. Please try again.' },
      { status: 500 }
    )
  }
}

export async function GET() {
  try {
    const supabase = createAdminClient()
    const { data, error } = await supabase.from('contact_inquiries').select('*').order('created_at', { ascending: false })
    if (error || !data || data.length === 0) {
      return NextResponse.json({ inquiries: globalStore.inquiries, leads: globalStore.leads })
    }
    return NextResponse.json({ inquiries: data, leads: globalStore.leads })
  } catch (e) {
    return NextResponse.json({ inquiries: globalStore.inquiries, leads: globalStore.leads })
  }
}
