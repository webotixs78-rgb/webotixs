import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function GET(request: NextRequest) {
  try {
    const supabase = createAdminClient()
    const { data: projects, error } = await supabase
      .from('crm_projects')
      .select(`
        *,
        client:crm_clients(*),
        tasks:crm_tasks(*),
        members:crm_project_members(*)
      `)
      .order('created_at', { ascending: false })

    if (error || !projects) {
      // Return empty or fallback structure if migration not yet applied locally
      return NextResponse.json({ projects: [] }, { status: 200 })
    }

    return NextResponse.json({ projects }, { status: 200 })
  } catch (e) {
    console.error('[CRM GET Projects Error]:', e)
    return NextResponse.json({ projects: [] }, { status: 200 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const {
      title,
      clientName,
      companyName,
      email,
      phone,
      website,
      packageType,
      budget,
      deadline,
      priority,
      workflowTemplateId,
      notes,
      requirements,
    } = body

    if (!title || !companyName || !email) {
      return NextResponse.json(
        { error: 'Project Title, Company Name, and Email are required.' },
        { status: 400 }
      )
    }

    const supabase = createAdminClient()

    // 1. Check or create Client (`crm_clients`)
    let clientId: string | null = null
    let clientCredentials: any = null

    const { data: existingClient } = await supabase
      .from('crm_clients')
      .select('id')
      .eq('email', email)
      .single()

    if (existingClient) {
      clientId = existingClient.id
    } else {
      const { data: newClient, error: clientErr } = await supabase
        .from('crm_clients')
        .insert({
          company_name: companyName,
          contact_name: clientName || companyName,
          email,
          phone: phone || null,
          website: website || null,
          status: 'active',
          notes: notes || null,
        })
        .select()
        .single()

      if (newClient) {
        clientId = newClient.id

        // Auto-generate Client Credentials (`crm_client_credentials`)
        const portalUsername = email.split('@')[0].replace(/[^a-zA-Z0-9]/g, '_')
        const tempPassword = `Webotixs!${companyName.replace(/[^a-zA-Z0-9]/g, '').slice(0, 6)}2026`
        const secretToken = `token_${Math.random().toString(36).slice(2, 10)}`

        const { data: credData } = await supabase
          .from('crm_client_credentials')
          .insert({
            client_id: clientId,
            portal_username: portalUsername,
            temp_password: tempPassword,
            secret_token: secretToken,
            is_active: true,
          })
          .select()
          .single()

        clientCredentials = credData || { portal_username: portalUsername, temp_password: tempPassword }

        // Optional: Send automated welcome email via Resend if configured
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
                from: 'Webotixs Agency Portal <portal@webotixs.com>',
                to: [email],
                subject: `Welcome to Webotixs Agency — Your Client Portal Credentials for ${companyName}`,
                html: `
                  <h2>Welcome to your Webotixs Client Dashboard</h2>
                  <p>Hi ${clientName || companyName},</p>
                  <p>Your new project <strong>${title}</strong> has been created. You can track real-time progress, review deliverables, and access invoices directly from your client portal.</p>
                  <hr/>
                  <p><strong>Portal Login:</strong> https://webotixs.com/admin/crm</p>
                  <p><strong>Username:</strong> ${portalUsername}</p>
                  <p><strong>Temporary Password:</strong> ${tempPassword}</p>
                  <p><strong>Secret Token:</strong> ${secretToken}</p>
                `,
              }),
            })
          }
        } catch (emailErr) {
          console.error('[Resend Email Error]:', emailErr)
        }
      }
    }

    if (!clientId) {
      return NextResponse.json(
        { error: 'Failed to create or retrieve client record.' },
        { status: 500 }
      )
    }

    // 2. Create Project (`crm_projects`)
    const { data: newProject, error: projErr } = await supabase
      .from('crm_projects')
      .insert({
        title,
        client_id: clientId,
        package_type: packageType || 'Custom Project',
        budget: parseFloat(budget) || 0.0,
        deadline: deadline || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        priority: priority || 'medium',
        status: 'In Progress',
        progress_percentage: 0,
        workflow_template_id: workflowTemplateId || null,
        notes: notes || null,
        requirements: requirements || null,
      })
      .select()
      .single()

    if (projErr || !newProject) {
      return NextResponse.json(
        { error: 'Failed to create project.', details: projErr?.message },
        { status: 500 }
      )
    }

    // 3. Auto-generate Workflow Tasks (`crm_tasks`) if template selected
    let generatedTasksCount = 0
    if (workflowTemplateId) {
      const { data: templateSteps } = await supabase
        .from('crm_workflow_steps')
        .select('*')
        .eq('template_id', workflowTemplateId)
        .order('step_order', { ascending: true })

      if (templateSteps && templateSteps.length > 0) {
        const tasksToInsert = templateSteps.map((step: any, index: number) => ({
          project_id: newProject.id,
          step_order: step.step_order,
          title: step.title,
          description: step.description,
          role_required: step.default_role,
          status: index === 0 ? 'Todo' : 'Locked', // Only Step 1 is unlocked initially
          due_date: new Date(Date.now() + (index + 1) * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          deliverable_type: step.required_deliverable || 'Deliverable URL',
        }))

        const { data: insertedTasks } = await supabase
          .from('crm_tasks')
          .insert(tasksToInsert)
          .select()

        generatedTasksCount = insertedTasks?.length || 0
      }
    }

    // 4. Log Activity (`crm_activity_logs`)
    await supabase.from('crm_activity_logs').insert({
      project_id: newProject.id,
      actor_name: 'Super Admin',
      actor_role: 'Super Admin',
      action: 'Project Created',
      details: `Created project "${title}" and auto-generated ${generatedTasksCount} workflow tasks. Client portal access active.`,
    })

    return NextResponse.json(
      {
        success: true,
        project: newProject,
        clientCredentials,
        generatedTasksCount,
      },
      { status: 201 }
    )
  } catch (e) {
    console.error('[CRM POST Project Error]:', e)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}
