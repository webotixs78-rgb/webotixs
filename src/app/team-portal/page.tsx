'use client'

import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { TeamDashboardView } from '@/components/admin/crm/TeamDashboardView'
import { ProjectDetailModal } from '@/components/admin/crm/ProjectDetailModal'
import { CRMTaskItem } from '@/components/admin/crm/KanbanBoard'
import { CRMRole } from '@/components/admin/crm/CRMRoleSwitcher'
import { Sparkles, Shield, AlertCircle, Briefcase, User, CheckCircle2 } from 'lucide-react'

// Clean initial arrays — zero demo content as requested by user
const seedProjects: any[] = []
const seedTasks: CRMTaskItem[] = []

function TeamPortalContent() {
  const searchParams = useSearchParams()
  const roleParam = (searchParams.get('role') as CRMRole) || 'UI/UX Designer'
  const emailParam = searchParams.get('email') || 'sarah@webotixs.com'

  const [projects, setProjects] = useState<any[]>(seedProjects)
  const [tasks, setTasks] = useState<CRMTaskItem[]>(seedTasks)
  const [selectedProject, setSelectedProject] = useState<any | null>(null)

  useEffect(() => {
    async function loadData() {
      try {
        const pRes = await fetch('/api/crm/projects').catch(() => null)
        let loadedApi = false
        if (pRes && pRes.ok) {
          const pData = await pRes.json()
          if (pData.projects && pData.projects.length > 0) {
            setProjects(pData.projects)
            loadedApi = true
            const allFetchedTasks: CRMTaskItem[] = []
            pData.projects.forEach((p: any) => {
              if (p.tasks) {
                p.tasks.forEach((t: any) => {
                  allFetchedTasks.push({
                    id: t.id,
                    project_id: p.id,
                    project_title: p.title,
                    step_order: t.step_order,
                    title: t.title,
                    description: t.description,
                    role_required: t.role_required,
                    status: t.status,
                    due_date: t.due_date,
                    deliverable_type: t.deliverable_type,
                    deliverable_url: t.deliverable_url,
                  })
                })
              }
            })
            if (allFetchedTasks.length > 0) setTasks(allFetchedTasks)
          }
        }

        if (!loadedApi) {
          const lp = localStorage.getItem('webotixs_crm_projects')
          const lt = localStorage.getItem('webotixs_crm_tasks')
          if (lp) setProjects(JSON.parse(lp))
          if (lt) setTasks(JSON.parse(lt))
        }
      } catch (err) {
        const lp = localStorage.getItem('webotixs_crm_projects')
        const lt = localStorage.getItem('webotixs_crm_tasks')
        if (lp) setProjects(JSON.parse(lp))
        if (lt) setTasks(JSON.parse(lt))
      }
    }
    loadData()
  }, [])

  const handleCompleteTask = async (taskId: string, deliverableUrl: string, notes: string) => {
    const target = tasks.find((t) => t.id === taskId)
    if (!target) return

    const pId = target.project_id
    const currentStep = target.step_order

    const updatedTasks = tasks.map((t) => {
      if (t.id === taskId) {
        return { ...t, status: 'Completed' as const, deliverable_url: deliverableUrl || t.deliverable_url }
      }
      if (t.project_id === pId && t.step_order === currentStep + 1 && t.status === 'Locked') {
        return { ...t, status: 'Todo' as const }
      }
      return t
    })

    setTasks(updatedTasks)
    localStorage.setItem('webotixs_crm_tasks', JSON.stringify(updatedTasks))
    
    // Also notify backend API if connected
    fetch('/api/crm/tasks/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId, deliverableUrl, notes }),
    }).catch(() => null)

    alert(`✅ Task Complete & Deliverable Submitted! Next pipeline task automatically unlocked.`)
  }

  const handleTaskStatusChange = (taskId: string, newStatus: CRMTaskItem['status']) => {
    const updated = tasks.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t))
    setTasks(updated)
    localStorage.setItem('webotixs_crm_tasks', JSON.stringify(updated))
  }

  return (
    <div className="space-y-8">
      {/* Staff Identity & Role Banner */}
      <div className="bg-gradient-to-r from-purple-900/40 via-[#0D1224] to-blue-900/30 border border-purple-500/30 rounded-3xl p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-glow-sm">
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-600 to-blue-500 flex items-center justify-center text-white font-display font-bold text-2xl shadow-lg shrink-0">
            {roleParam.substring(0, 2).toUpperCase()}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 text-xs font-bold uppercase tracking-wider">
                Assigned Department
              </span>
            </div>
            <h1 className="font-display text-xl md:text-2xl font-bold text-white tracking-tight mt-1.5">
              {roleParam} Pipeline Board
            </h1>
            <p className="text-xs md:text-sm text-[#94A3B8] flex items-center gap-2 mt-1">
              <User size={13} className="text-purple-400" /> Logged in staff account: <strong className="text-white">{emailParam}</strong>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-[#050816]/80 px-4 py-3 rounded-2xl border border-[#273449]">
          <Briefcase size={18} className="text-purple-400 shrink-0" />
          <div className="text-xs">
            <div className="text-white font-semibold">Department Task Pipeline</div>
            <div className="text-[#94A3B8]">Filtered only for {roleParam} deliverables</div>
          </div>
        </div>
      </div>

      {/* Render TeamDashboardView cleanly */}
      <TeamDashboardView
        tasks={tasks}
        projects={projects}
        currentRole={roleParam}
        onOpenTaskDetail={(t) => {
          const p = projects.find((proj) => proj.id === t.project_id)
          if (p) setSelectedProject(p)
        }}
        onTaskStatusChange={handleTaskStatusChange}
        onCompleteTask={handleCompleteTask}
      />

      {/* Project Command Modal */}
      {selectedProject && (
        <ProjectDetailModal
          project={selectedProject}
          tasks={tasks}
          currentRole={roleParam}
          onClose={() => setSelectedProject(null)}
          onCompleteTask={handleCompleteTask}
        />
      )}
    </div>
  )
}

export const dynamic = 'force-dynamic'

export default function TeamPortalPage() {
  return (
    <React.Suspense fallback={<div className="text-center py-20 text-[#94A3B8]">Loading Team Workspace...</div>}>
      <TeamPortalContent />
    </React.Suspense>
  )
}
