import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { taskId, projectId, actorName, actorRole, deliverableUrl, deliverableNotes } = body

    if (!taskId || !projectId) {
      return NextResponse.json(
        { error: 'Task ID and Project ID are required.' },
        { status: 400 }
      )
    }

    const supabase = createAdminClient()

    // 1. Fetch current task
    const { data: task, error: taskErr } = await supabase
      .from('crm_tasks')
      .select('*')
      .eq('id', taskId)
      .single()

    if (taskErr || !task) {
      return NextResponse.json({ error: 'Task not found.' }, { status: 404 })
    }

    // 2. Mark task as completed
    const { error: updateErr } = await supabase
      .from('crm_tasks')
      .update({
        status: 'Completed',
        deliverable_url: deliverableUrl || task.deliverable_url || null,
        deliverable_notes: deliverableNotes || null,
        completed_at: new Date().toISOString(),
      })
      .eq('id', taskId)

    if (updateErr) {
      return NextResponse.json(
        { error: 'Failed to complete task.', details: updateErr.message },
        { status: 500 }
      )
    }

    // 3. Find all tasks for this project to check dependencies and recalculate progress
    const { data: allTasks } = await supabase
      .from('crm_tasks')
      .select('*')
      .eq('project_id', projectId)
      .order('step_order', { ascending: true })

    let unlockedTaskTitle = null
    let newProgress = 100

    if (allTasks && allTasks.length > 0) {
      const currentStep = task.step_order
      const nextTask = allTasks.find((t: any) => t.step_order === currentStep + 1)

      // Unlock next task if locked
      if (nextTask && nextTask.status === 'Locked') {
        await supabase
          .from('crm_tasks')
          .update({ status: 'Todo' })
          .eq('id', nextTask.id)

        unlockedTaskTitle = nextTask.title
      }

      // Calculate overall progress percentage
      const completedCount = allTasks.filter(
        (t: any) => t.id === taskId || t.status === 'Completed'
      ).length
      newProgress = Math.round((completedCount / allTasks.length) * 100)

      // Update Project status and progress percentage
      const projectStatus = newProgress === 100 ? 'Review' : 'In Progress'
      await supabase
        .from('crm_projects')
        .update({
          progress_percentage: newProgress,
          status: projectStatus,
          updated_at: new Date().toISOString(),
        })
        .eq('id', projectId)
    }

    // 4. Log Activity (`crm_activity_logs`)
    const logDetails = unlockedTaskTitle
      ? `Completed task "${task.title}". Automatically unlocked next step: "${unlockedTaskTitle}". Project progress now at ${newProgress}%.`
      : `Completed task "${task.title}". Project progress now at ${newProgress}%.`

    await supabase.from('crm_activity_logs').insert({
      project_id: projectId,
      actor_name: actorName || 'Team Member',
      actor_role: actorRole || task.role_required || 'Team Member',
      action: 'Task Completed',
      details: logDetails,
    })

    // 5. Send Notification
    await supabase.from('crm_notifications').insert({
      title: `Task Completed: ${task.title}`,
      message: `${actorName || 'Team Member'} marked "${task.title}" as completed. ${
        unlockedTaskTitle ? `Step "${unlockedTaskTitle}" is now unlocked and ready.` : ''
      }`,
      type: 'task',
    })

    return NextResponse.json(
      {
        success: true,
        progress: newProgress,
        unlockedTask: unlockedTaskTitle,
      },
      { status: 200 }
    )
  } catch (e) {
    console.error('[CRM Task Complete Error]:', e)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}
