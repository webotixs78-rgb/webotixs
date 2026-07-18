import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'
import { contactSchema } from '@/lib/validations/contact'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // Validate with Zod
    const parsed = contactSchema.safeParse(body)
    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Validation failed', details: parsed.error.flatten().fieldErrors },
        { status: 400 }
      )
    }

    const { name, email, phone, company, service, budget, message } = parsed.data

    // AI Lead Classification (GPT-4o-mini)
    let ai_priority: 'high' | 'medium' | 'low' = 'medium'
    let ai_summary = ''

    try {
      const openaiKey = process.env.OPENAI_API_KEY
      if (openaiKey) {
        const aiResponse = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${openaiKey}`,
          },
          body: JSON.stringify({
            model: 'gpt-4o-mini',
            messages: [
              {
                role: 'system',
                content: `You are an AI lead classifier for a web design agency called Webotixs. Analyze the contact form submission and return a JSON object with two fields:
- "priority": "high", "medium", or "low" based on budget, project scope, company size, and urgency
- "summary": A brief 1-2 sentence summary of the lead quality and recommended action

Rules:
- Budget $50,000+ or enterprise clients = high
- Budget $10,000-$50,000 or clear project scope = medium  
- Budget under $10,000, vague requests, or spam = low

Return ONLY valid JSON, no markdown.`,
              },
              {
                role: 'user',
                content: `Name: ${name}\nEmail: ${email}\nCompany: ${company || 'N/A'}\nService: ${service || 'N/A'}\nBudget: ${budget || 'N/A'}\nMessage: ${message}`,
              },
            ],
            temperature: 0.3,
            max_tokens: 150,
          }),
        })

        if (aiResponse.ok) {
          const aiData = await aiResponse.json()
          const content = aiData.choices?.[0]?.message?.content?.trim()
          if (content) {
            const classification = JSON.parse(content)
            ai_priority = classification.priority || 'medium'
            ai_summary = classification.summary || ''
          }
        }
      }
    } catch (aiError) {
      // AI classification is optional — continue without it
      console.error('[AI Classification Error]:', aiError)
      ai_summary = 'AI classification unavailable. Manual review required.'
    }

    // Insert into Supabase
    const supabase = createAdminClient()
    const { data, error } = await supabase
      .from('contact_submissions')
      .insert({
        name,
        email,
        phone: phone || null,
        company: company || null,
        service: service || null,
        budget: budget || null,
        message,
        ai_priority,
        ai_summary,
        status: 'new',
      })
      .select()
      .single()

    if (error) {
      console.error('[Supabase Insert Error]:', error)
      return NextResponse.json(
        { error: 'Failed to save submission. Please try again.' },
        { status: 500 }
      )
    }

    // Optional: Send notification email via Resend
    try {
      const resendKey = process.env.RESEND_API_KEY
      const notifyEmail = process.env.NOTIFICATION_EMAIL
      if (resendKey && notifyEmail) {
        await fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${resendKey}`,
          },
          body: JSON.stringify({
            from: 'Webotixs CRM <noreply@webotixs.com>',
            to: [notifyEmail],
            subject: `[${ai_priority.toUpperCase()}] New Lead: ${name} — ${service || 'General'}`,
            html: `
              <h2>New Contact Submission</h2>
              <p><strong>Name:</strong> ${name}</p>
              <p><strong>Email:</strong> ${email}</p>
              <p><strong>Phone:</strong> ${phone || 'N/A'}</p>
              <p><strong>Company:</strong> ${company || 'N/A'}</p>
              <p><strong>Service:</strong> ${service || 'N/A'}</p>
              <p><strong>Budget:</strong> ${budget || 'N/A'}</p>
              <p><strong>Message:</strong> ${message}</p>
              <hr/>
              <p><strong>AI Priority:</strong> ${ai_priority}</p>
              <p><strong>AI Summary:</strong> ${ai_summary}</p>
            `,
          }),
        })
      }
    } catch (emailError) {
      console.error('[Email Notification Error]:', emailError)
    }

    return NextResponse.json(
      { success: true, id: data?.id, ai_priority, ai_summary },
      { status: 201 }
    )
  } catch (e) {
    console.error('[Contact API Error]:', e)
    return NextResponse.json(
      { error: 'Internal server error.' },
      { status: 500 }
    )
  }
}
