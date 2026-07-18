import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function GET(request: NextRequest) {
  try {
    const supabase = createAdminClient()
    const { data: invoices, error } = await supabase
      .from('crm_invoices')
      .select(`
        *,
        client:crm_clients(company_name, contact_name, email),
        project:crm_projects(title)
      `)
      .order('issue_date', { ascending: false })

    if (error || !invoices) {
      return NextResponse.json({ invoices: [] }, { status: 200 })
    }

    return NextResponse.json({ invoices }, { status: 200 })
  } catch (e) {
    console.error('[CRM GET Invoices Error]:', e)
    return NextResponse.json({ invoices: [] }, { status: 200 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { invoiceNumber, projectId, clientId, amount, taxRate, issueDate, dueDate, notes, status } = body

    if (!invoiceNumber || !clientId || !amount) {
      return NextResponse.json(
        { error: 'Invoice Number, Client ID, and Amount are required.' },
        { status: 400 }
      )
    }

    const numericAmount = parseFloat(amount) || 0.0
    const numericTax = numericAmount * (parseFloat(taxRate) || 0.05)
    const totalAmount = numericAmount + numericTax

    const supabase = createAdminClient()
    const { data: newInvoice, error } = await supabase
      .from('crm_invoices')
      .insert({
        invoice_number: invoiceNumber,
        project_id: projectId || null,
        client_id: clientId,
        amount: numericAmount,
        tax_amount: numericTax,
        total_amount: totalAmount,
        status: status || 'Pending',
        issue_date: issueDate || new Date().toISOString().split('T')[0],
        due_date: dueDate || new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        notes: notes || null,
      })
      .select()
      .single()

    if (error || !newInvoice) {
      return NextResponse.json(
        { error: 'Failed to create invoice.', details: error?.message },
        { status: 500 }
      )
    }

    return NextResponse.json({ success: true, invoice: newInvoice }, { status: 201 })
  } catch (e) {
    console.error('[CRM POST Invoice Error]:', e)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}
