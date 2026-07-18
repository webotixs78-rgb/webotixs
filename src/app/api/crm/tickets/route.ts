import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function GET(request: NextRequest) {
  try {
    const supabase = createAdminClient()
    const { data: tickets, error } = await supabase
      .from('crm_support_tickets')
      .select(`
        *,
        client:crm_clients(company_name, contact_name, email),
        project:crm_projects(title)
      `)
      .order('created_at', { ascending: false })

    if (error || !tickets) {
      return NextResponse.json({ tickets: [] }, { status: 200 })
    }

    return NextResponse.json({ tickets }, { status: 200 })
  } catch (e) {
    console.error('[CRM GET Tickets Error]:', e)
    return NextResponse.json({ tickets: [] }, { status: 200 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { ticketNumber, clientId, projectId, subject, description, priority } = body

    if (!subject || !description || !clientId) {
      return NextResponse.json(
        { error: 'Subject, Description, and Client ID are required.' },
        { status: 400 }
      )
    }

    const supabase = createAdminClient()
    const finalTicketNumber = ticketNumber || `TICK-${Math.floor(100 + Math.random() * 900)}`

    const { data: newTicket, error } = await supabase
      .from('crm_support_tickets')
      .insert({
        ticket_number: finalTicketNumber,
        client_id: clientId,
        project_id: projectId || null,
        subject,
        description,
        priority: priority || 'medium',
        status: 'Open',
      })
      .select()
      .single()

    if (error || !newTicket) {
      return NextResponse.json(
        { error: 'Failed to create support ticket.', details: error?.message },
        { status: 500 }
      )
    }

    // Notify Admin of new support ticket
    await supabase.from('crm_notifications').insert({
      title: `New Support Ticket: ${finalTicketNumber}`,
      message: `Client opened support ticket "${subject}". Priority: ${priority || 'medium'}.`,
      type: 'info',
    })

    return NextResponse.json({ success: true, ticket: newTicket }, { status: 201 })
  } catch (e) {
    console.error('[CRM POST Ticket Error]:', e)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { ticketId, status, assignedTo } = body

    if (!ticketId || !status) {
      return NextResponse.json(
        { error: 'Ticket ID and Status are required.' },
        { status: 400 }
      )
    }

    const supabase = createAdminClient()
    const { data: updatedTicket, error } = await supabase
      .from('crm_support_tickets')
      .update({
        status,
        assigned_to: assignedTo || null,
        updated_at: new Date().toISOString(),
      })
      .eq('id', ticketId)
      .select()
      .single()

    if (error) {
      return NextResponse.json(
        { error: 'Failed to update ticket.', details: error.message },
        { status: 500 }
      )
    }

    return NextResponse.json({ success: true, ticket: updatedTicket }, { status: 200 })
  } catch (e) {
    console.error('[CRM PUT Ticket Error]:', e)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}
