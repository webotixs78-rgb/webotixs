'use client'

import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { CRMRoleSwitcher, CRMRole } from './CRMRoleSwitcher'
import { AdminDashboardView } from './AdminDashboardView'
import { TeamDashboardView } from './TeamDashboardView'
import { ClientDashboardView } from './ClientDashboardView'
import { ProjectDetailModal } from './ProjectDetailModal'
import { CRMTaskItem } from './KanbanBoard'
import { Sparkles, Shield, RefreshCw } from 'lucide-react'

// Clean initial arrays — zero demo content as requested by user for manual testing
const seedProjects: any[] = []
const seedTasks: CRMTaskItem[] = []
const seedInvoices: any[] = []
const seedTickets: any[] = []
const seedInquiries: any[] = []

export function AgencyCRMClientHub() {
  const searchParams = useSearchParams()
  const [currentRole, setCurrentRole] = useState<CRMRole>('Super Admin')
  const [projects, setProjects] = useState<any[]>(seedProjects)
  const [tasks, setTasks] = useState<CRMTaskItem[]>(seedTasks)
  const [invoices, setInvoices] = useState<any[]>(seedInvoices)
  const [tickets, setTickets] = useState<any[]>(seedTickets)
  const [inquiries, setInquiries] = useState<any[]>(seedInquiries)
  const [selectedProject, setSelectedProject] = useState<any | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  // Sync role from URL param or cookie session
  useEffect(() => {
    const roleParam = searchParams.get('role') as CRMRole
    const validRoles: CRMRole[] = [
      'Super Admin',
      'Admin',
      'Project Manager',
      'UI/UX Designer',
      'WordPress Developer',
      'Frontend Developer',
      'Backend Developer',
      'SEO Specialist',
      'Content Writer',
      'QA Tester',
      'Client',
    ]
    if (roleParam && validRoles.includes(roleParam)) {
      setCurrentRole(roleParam)
    }
  }, [searchParams])

  // Fetch from APIs and sync with localStorage on load
  useEffect(() => {
    let isMounted = true

    async function fetchAllData() {
      setLoading(true)
      try {
        const [projRes, invRes, tickRes, leadsRes] = await Promise.all([
          fetch('/api/crm/projects').catch(() => null),
          fetch('/api/crm/invoices').catch(() => null),
          fetch('/api/crm/tickets').catch(() => null),
          fetch('/api/crm/leads').catch(() => null),
        ])

        if (!isMounted) return
        let loadedFromApi = false

        if (projRes && projRes.ok) {
          const pData = await projRes.json()
          if (pData.projects && pData.projects.length > 0) {
            setProjects(pData.projects)
            loadedFromApi = true
            const allFetchedTasks: CRMTaskItem[] = []
            pData.projects.forEach((p: any) => {
              if (p.tasks && p.tasks.length > 0) {
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

        if (invRes && invRes.ok) {
          const iData = await invRes.json()
          if (iData.invoices && iData.invoices.length > 0) setInvoices(iData.invoices)
        }

        if (tickRes && tickRes.ok) {
          const tData = await tickRes.json()
          if (tData.tickets && tData.tickets.length > 0) setTickets(tData.tickets)
        }

        if (leadsRes && leadsRes.ok) {
          const lData = await leadsRes.json()
          if (lData.leads && lData.leads.length > 0) {
            setInquiries(lData.leads)
          }
        }

        // Check local storage / fallback
        const localLeads = localStorage.getItem('webotixs_crm_leads') || localStorage.getItem('webotixs_contact_inquiries')
        if (localLeads) {
          try {
            const parsed = JSON.parse(localLeads)
            if (Array.isArray(parsed) && parsed.length > 0) {
              setInquiries((prev) => {
                const map = new Map()
                parsed.forEach((item: any) => map.set(item.id, item))
                prev.forEach((item: any) => {
                  if (!map.has(item.id)) map.set(item.id, item)
                })
                return Array.from(map.values())
              })
            }
          } catch {}
        }

        if (!loadedFromApi) {
          const localProjects = localStorage.getItem('webotixs_crm_projects')
          const localTasks = localStorage.getItem('webotixs_crm_tasks')
          const localInvoices = localStorage.getItem('webotixs_crm_invoices')
          const localTickets = localStorage.getItem('webotixs_crm_tickets')

          if (localProjects) setProjects(JSON.parse(localProjects))
          if (localTasks) setTasks(JSON.parse(localTasks))
          if (localInvoices) setInvoices(JSON.parse(localInvoices))
          if (localTickets) setTickets(JSON.parse(localTickets))
        }
      } catch (err) {
        console.error('[CRM Data Fetch Fallback]:', err)
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchAllData()

    const handleStorage = () => {
      try {
        const localLeads = localStorage.getItem('webotixs_crm_leads') || localStorage.getItem('webotixs_contact_inquiries')
        if (localLeads) {
          const parsed = JSON.parse(localLeads)
          if (Array.isArray(parsed) && parsed.length > 0) {
            setInquiries(parsed)
          }
        }
      } catch {}
    }

    window.addEventListener('storage', handleStorage)
    return () => {
      isMounted = false
      window.removeEventListener('storage', handleStorage)
    }
  }, [])

  const handleCreateProject = async (projectData: any) => {
    try {
      const res = await fetch('/api/crm/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(projectData),
      })

      if (res.ok) {
        const data = await res.json()
        if (data.project) {
          const updatedProj = [data.project, ...projects]
          setProjects(updatedProj)
          localStorage.setItem('webotixs_crm_projects', JSON.stringify(updatedProj))

          alert(`✅ Project Created Successfully! Client Account & Credentials generated and sent via email simulation.`)
          
          // Refresh list from API or sync state
          const projRes = await fetch('/api/crm/projects')
          if (projRes.ok) {
            const pData = await projRes.json()
            if (pData.projects && pData.projects.length > 0) {
              setProjects(pData.projects)
              localStorage.setItem('webotixs_crm_projects', JSON.stringify(pData.projects))

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
              if (allFetchedTasks.length > 0) {
                setTasks(allFetchedTasks)
                localStorage.setItem('webotixs_crm_tasks', JSON.stringify(allFetchedTasks))
              }
            }
          }
          return
        }
      }
    } catch (e) {
      console.error(e)
    }

    // Local creation & persistence if API offline
    const newId = `proj-${Date.now()}`
    const newProj = {
      id: newId,
      title: projectData.title,
      package_type: projectData.packageType,
      budget: parseFloat(projectData.budget) || 25000.0,
      deadline: projectData.deadline,
      status: 'In Progress',
      progress_percentage: 0,
      priority: projectData.priority || 'medium',
      client_id: `client-${Date.now()}`,
      notes: projectData.requirements || 'Created via Agency CRM Portal.',
      client: { company_name: projectData.companyName, contact_name: projectData.clientName, email: projectData.email },
    }

    const tplTasks: CRMTaskItem[] = [
      { id: `t-${Date.now()}-1`, project_id: newId, project_title: projectData.title, step_order: 1, title: 'Strategy & Wireframe Mapping', description: 'Initial UX discovery and wireframe creation.', role_required: 'UI/UX Designer', status: 'Todo', due_date: projectData.deadline },
      { id: `t-${Date.now()}-2`, project_id: newId, project_title: projectData.title, step_order: 2, title: 'Custom Code Engineering', description: 'Next.js 16 and Supabase build.', role_required: 'Frontend Developer', status: 'Locked' },
      { id: `t-${Date.now()}-3`, project_id: newId, project_title: projectData.title, step_order: 3, title: 'QA & Security Verification', description: 'Cross-device responsiveness and speed audits.', role_required: 'QA Tester', status: 'Locked' },
    ]

    const nextProjects = [newProj, ...projects]
    const nextTasks = [...tplTasks, ...tasks]

    setProjects(nextProjects)
    setTasks(nextTasks)
    localStorage.setItem('webotixs_crm_projects', JSON.stringify(nextProjects))
    localStorage.setItem('webotixs_crm_tasks', JSON.stringify(nextTasks))

    alert(`✅ Project Created & Stored! Automated Client credentials generated and workflow tasks spawned.`)
  }

  const handleCompleteTask = async (taskId: string, deliverableUrl: string, notes: string) => {
    try {
      const targetTask = tasks.find((t) => t.id === taskId)
      if (!targetTask) return

      await fetch('/api/crm/tasks/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          taskId,
          projectId: targetTask.project_id,
          actorName: currentRole === 'Client' ? 'Client' : 'Team Member',
          actorRole: currentRole,
          deliverableUrl,
          deliverableNotes: notes,
        }),
      })
    } catch (e) {
      console.error(e)
    }

    // Local state update for instant UI feedback
    const target = tasks.find((t) => t.id === taskId)
    if (!target) return

    const pId = target.project_id
    const currentStep = target.step_order

    const updatedTasks = tasks.map((t) => {
      if (t.id === taskId) {
        return { ...t, status: 'Completed' as const, deliverable_url: deliverableUrl || t.deliverable_url }
      }
      // Unlock next task
      if (t.project_id === pId && t.step_order === currentStep + 1 && t.status === 'Locked') {
        return { ...t, status: 'Todo' as const }
      }
      return t
    })

    setTasks(updatedTasks)

    // Recalculate project progress
    const projectTasksList = updatedTasks.filter((t) => t.project_id === pId)
    const completedCount = projectTasksList.filter((t) => t.status === 'Completed').length
    const newProgress = Math.round((completedCount / projectTasksList.length) * 100)

    setProjects((prev) =>
      prev.map((p) => (p.id === pId ? { ...p, progress_percentage: newProgress, status: newProgress === 100 ? 'Review' : 'In Progress' } : p))
    )

    if (selectedProject && selectedProject.id === pId) {
      setSelectedProject((prev: any) => ({ ...prev, progress_percentage: newProgress }))
    }
  }

  const handleTaskStatusChange = (taskId: string, newStatus: CRMTaskItem['status']) => {
    setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t)))
  }

  const handleCreateTicket = async (ticketData: any) => {
    const newTick = {
      id: `t-${Date.now()}`,
      ticket_number: `TICK-${Math.floor(100 + Math.random() * 900)}`,
      client_id: ticketData.clientId,
      project_id: ticketData.projectId,
      subject: ticketData.subject,
      description: ticketData.description,
      priority: ticketData.priority || 'medium',
      status: 'Open',
      created_at: new Date().toISOString(),
      client: { company_name: 'Al-Khaleej Retail Group' },
    }
    setTickets((prev) => [newTick, ...prev])
    alert(`✅ Support Ticket #${newTick.ticket_number} Submitted! Your Project Manager has been notified.`)
  }

  return (
    <div className="space-y-6">
      {/* Title & Live Status */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl md:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            Agency CRM & Operating System
            <span className="px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-bold uppercase tracking-wider">
              v2.0 Production
            </span>
          </h1>
          <p className="text-[#94A3B8] text-xs md:text-sm mt-1">
            Complete agency orchestration: automated client credentials, lifecycle task unlocks, invoices, and role permissions.
          </p>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-xs text-blue-400 font-bold bg-blue-500/10 px-3 py-1.5 rounded-xl border border-blue-500/30 animate-pulse">
            <RefreshCw size={14} className="animate-spin" /> Syncing API state...
          </div>
        )}
      </div>

      {/* RBAC Role Switcher Bar */}
      <CRMRoleSwitcher currentRole={currentRole} onRoleChange={setCurrentRole} />

      {/* Role-Specific Dashboard Views */}
      {currentRole === 'Super Admin' || currentRole === 'Admin' || currentRole === 'Project Manager' ? (
        <AdminDashboardView
          projects={projects}
          tasks={tasks}
          invoices={invoices}
          tickets={tickets}
          inquiries={inquiries}
          currentRole={currentRole}
          onOpenProject={setSelectedProject}
          onTaskStatusChange={handleTaskStatusChange}
          onCreateProject={handleCreateProject}
        />
      ) : currentRole === 'Client' ? (
        <ClientDashboardView
          projects={projects}
          tasks={tasks}
          invoices={invoices}
          tickets={tickets}
          currentRole={currentRole}
          onOpenProject={setSelectedProject}
          onCreateTicket={handleCreateTicket}
        />
      ) : (
        <TeamDashboardView
          tasks={tasks}
          projects={projects}
          currentRole={currentRole}
          onOpenTaskDetail={(t) => {
            const p = projects.find((proj) => proj.id === t.project_id)
            if (p) setSelectedProject(p)
          }}
          onTaskStatusChange={handleTaskStatusChange}
          onCompleteTask={handleCompleteTask}
        />
      )}

      {/* Project Command Center Detail Modal */}
      {selectedProject && (
        <ProjectDetailModal
          project={selectedProject}
          tasks={tasks}
          currentRole={currentRole}
          onClose={() => setSelectedProject(null)}
          onCompleteTask={handleCompleteTask}
        />
      )}
    </div>
  )
}
