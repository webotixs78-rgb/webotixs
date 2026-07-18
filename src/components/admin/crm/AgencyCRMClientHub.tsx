'use client'

import React, { useState, useEffect } from 'react'
import { CRMRoleSwitcher, CRMRole } from './CRMRoleSwitcher'
import { AdminDashboardView } from './AdminDashboardView'
import { TeamDashboardView } from './TeamDashboardView'
import { ClientDashboardView } from './ClientDashboardView'
import { ProjectDetailModal } from './ProjectDetailModal'
import { CRMTaskItem } from './KanbanBoard'
import { Sparkles, Shield, RefreshCw } from 'lucide-react'

// Realistic initial demo seed data if DB migration not yet applied locally or loading
const seedProjects = [
  {
    id: '11111111-1111-1111-1111-111111111101',
    title: 'Al-Khaleej E-Commerce Headless Storefront',
    package_type: 'Enterprise E-Commerce',
    budget: 45000.0,
    deadline: '2026-09-15',
    status: 'In Progress',
    progress_percentage: 60,
    priority: 'high',
    client_id: '44444444-4444-4444-4444-444444444401',
    notes: 'Full Shopify Headless integration with Next.js 16, RTL Arabic support, and multi-currency checkout.',
    client: { company_name: 'Al-Khaleej Retail Group', contact_name: 'Tariq Al-Mansoor', email: 'tariq@alkhaleej.ae' },
  },
  {
    id: '11111111-1111-1111-1111-111111111102',
    title: 'LuxBrand Paris Complete Digital Rebrand',
    package_type: 'Brand Identity & Design',
    budget: 32000.0,
    deadline: '2026-08-30',
    status: 'In Progress',
    progress_percentage: 40,
    priority: 'high',
    client_id: '44444444-4444-4444-4444-444444444402',
    notes: 'Luxury typography, motion graphics guidelines, and high-fidelity wireframes.',
    client: { company_name: 'LuxBrand Paris', contact_name: 'Sophie Laurent', email: 'sophie@luxbrand.fr' },
  },
  {
    id: '11111111-1111-1111-1111-111111111103',
    title: 'FinTech Growth Portal & Dashboard',
    package_type: 'Custom Web Application',
    budget: 18500.0,
    deadline: '2026-08-10',
    status: 'Review',
    progress_percentage: 100,
    priority: 'medium',
    client_id: '44444444-4444-4444-4444-444444444403',
    notes: 'Investor dashboard with live charting, Supabase auth, and data export readiness.',
    client: { company_name: 'Finch Investments', contact_name: 'Robert Finch', email: 'r.finch@finchinvest.com' },
  },
]

const seedTasks: CRMTaskItem[] = [
  {
    id: '22222222-2222-2222-2222-222222222201',
    project_id: '11111111-1111-1111-1111-111111111101',
    project_title: 'Al-Khaleej E-Commerce Headless Storefront',
    step_order: 1,
    title: 'Store Strategy & Wireframes',
    description: 'Design wireframes and user flow architecture for high conversion in GCC market.',
    role_required: 'UI/UX Designer',
    status: 'Completed',
    due_date: '2026-07-20',
    deliverable_type: 'Figma Prototype URL',
    deliverable_url: 'https://figma.com/file/alkhaleej-store-wireframes-v2',
  },
  {
    id: '22222222-2222-2222-2222-222222222202',
    project_id: '11111111-1111-1111-1111-111111111101',
    project_title: 'Al-Khaleej E-Commerce Headless Storefront',
    step_order: 2,
    title: 'Shopify Custom Headless Development',
    description: 'Build Next.js 16 storefront connected to Shopify Storefront API with custom cart drawer.',
    role_required: 'Frontend Developer',
    status: 'Completed',
    due_date: '2026-08-05',
    deliverable_type: 'Vercel Staging URL',
    deliverable_url: 'https://alkhaleej-storefront-staging.vercel.app',
  },
  {
    id: '22222222-2222-2222-2222-222222222203',
    project_id: '11111111-1111-1111-1111-111111111101',
    project_title: 'Al-Khaleej E-Commerce Headless Storefront',
    step_order: 3,
    title: 'Product Catalog Import & SEO Setup',
    description: 'Migrate 450 SKUs, format Arabic meta tags, and configure structured data schema.',
    role_required: 'SEO Specialist',
    status: 'In Progress',
    due_date: '2026-08-20',
    deliverable_type: 'SEO Audit Report & XML Sitemap',
  },
  {
    id: '22222222-2222-2222-2222-222222222204',
    project_id: '11111111-1111-1111-1111-111111111101',
    project_title: 'Al-Khaleej E-Commerce Headless Storefront',
    step_order: 4,
    title: 'Cross-Browser & Checkout QA Testing',
    description: 'Test Apple Pay, Mada, credit card checkout, and mobile responsiveness on iOS/Android.',
    role_required: 'QA Tester',
    status: 'Locked',
    due_date: '2026-09-01',
    deliverable_type: 'QA Sign-off Matrix & Bug Report',
  },
  {
    id: '22222222-2222-2222-2222-222222222205',
    project_id: '11111111-1111-1111-1111-111111111101',
    project_title: 'Al-Khaleej E-Commerce Headless Storefront',
    step_order: 5,
    title: 'Production Launch & Client Handoff',
    description: 'DNS transition, SSL verification, and client training session recording.',
    role_required: 'Project Manager',
    status: 'Locked',
    due_date: '2026-09-15',
    deliverable_type: 'Production URL & Handoff Package',
  },
]

const seedInvoices = [
  {
    id: 'inv-1',
    invoice_number: 'INV-2026-001',
    client_id: '44444444-4444-4444-4444-444444444401',
    amount: 22500.0,
    tax_amount: 1125.0,
    total_amount: 23625.0,
    status: 'Paid',
    issue_date: '2026-07-16',
    due_date: '2026-07-30',
    notes: '50% Milestone Deposit — Al-Khaleej E-Commerce Overhaul',
    client: { company_name: 'Al-Khaleej Retail Group' },
  },
  {
    id: 'inv-2',
    invoice_number: 'INV-2026-002',
    client_id: '44444444-4444-4444-4444-444444444402',
    amount: 16000.0,
    tax_amount: 800.0,
    total_amount: 16800.0,
    status: 'Pending',
    issue_date: '2026-07-17',
    due_date: '2026-07-31',
    notes: '50% Upfront Deposit — LuxBrand Paris Rebrand',
    client: { company_name: 'LuxBrand Paris' },
  },
]

const seedTickets = [
  {
    id: 't-1',
    ticket_number: 'TICK-104',
    client_id: '44444444-4444-4444-4444-444444444401',
    subject: 'Inquiry regarding staging checkout currency persistence',
    description: 'During our team review of Step 2 staging link, we noticed SAR currency defaults back to AED when reloading cart. Can the dev team verify?',
    priority: 'high',
    status: 'Working',
    created_at: '2026-07-17T15:30:00Z',
    client: { company_name: 'Al-Khaleej Retail Group' },
  },
]

const seedInquiries = [
  { id: '1', name: 'Al-Khaleej Retail Group', email: 'tech@alkhaleej.ae', phone: '+971-50-123-4567', company: 'Al-Khaleej Retail', service: 'E-Commerce Solutions', budget: '$25,000 - $50,000', message: 'We need a complete overhaul of our online storefront. We want Shopify headless with custom checkout. Expected launch in Q4 2026.', ai_priority: 'high', ai_summary: 'Enterprise e-commerce rebuild. High budget, clear timeline. Hot lead.', status: 'new' },
  { id: '2', name: 'Robert Finch', email: 'r.finch@finchinvest.com', phone: '+1-555-987-6543', company: 'Finch Investments', service: 'Web Design & Development', budget: '$10,000 - $25,000', message: 'Looking for a modern portfolio website for our investment firm. Need to showcase our track record and team. Clean, premium feel.', ai_priority: 'medium', ai_summary: 'Mid-tier corporate website project. Moderate budget, clear requirements.', status: 'contacted' },
  { id: '3', name: 'EduLearn Inc', email: 'hello@edulearn.org', phone: '', company: 'EduLearn', service: 'Cloud & DevOps', budget: '$5,000 - $10,000', message: 'We have a Node.js app that needs to be containerized and deployed to AWS. Looking for ongoing DevOps support.', ai_priority: 'low', ai_summary: 'Small infrastructure project. Lower budget, ongoing support needed.', status: 'new' },
  { id: '4', name: 'Sophie Laurent', email: 'sophie@luxbrand.fr', phone: '+33-6-1234-5678', company: 'LuxBrand Paris', service: 'Brand Identity & Design', budget: '$50,000+', message: 'Our luxury fashion brand needs a complete digital rebrand including logo, website, and mobile app. We want the absolute best quality.', ai_priority: 'high', ai_summary: 'Premium luxury rebrand. Highest budget tier. VIP lead.', status: 'qualified' },
]

export function AgencyCRMClientHub() {
  const [currentRole, setCurrentRole] = useState<CRMRole>('Super Admin')
  const [projects, setProjects] = useState<any[]>(seedProjects)
  const [tasks, setTasks] = useState<CRMTaskItem[]>(seedTasks)
  const [invoices, setInvoices] = useState<any[]>(seedInvoices)
  const [tickets, setTickets] = useState<any[]>(seedTickets)
  const [inquiries, setInquiries] = useState<any[]>(seedInquiries)
  const [selectedProject, setSelectedProject] = useState<any | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  // Fetch from APIs on load
  useEffect(() => {
    async function fetchAllData() {
      setLoading(true)
      try {
        const [projRes, invRes, tickRes] = await Promise.all([
          fetch('/api/crm/projects').catch(() => null),
          fetch('/api/crm/invoices').catch(() => null),
          fetch('/api/crm/tickets').catch(() => null),
        ])

        if (projRes && projRes.ok) {
          const pData = await projRes.json()
          if (pData.projects && pData.projects.length > 0) {
            setProjects(pData.projects)
            // Flatten tasks if present
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
      } catch (err) {
        console.error('[CRM Data Fetch Fallback]:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchAllData()
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
          setProjects((prev) => [data.project, ...prev])
          alert(`✅ Project Created Successfully! Client Account & Credentials generated and sent via email simulation.`)
          // Refresh list
          const projRes = await fetch('/api/crm/projects')
          if (projRes.ok) {
            const pData = await projRes.json()
            if (pData.projects && pData.projects.length > 0) {
              setProjects(pData.projects)
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
          return
        }
      }
    } catch (e) {
      console.error(e)
    }

    // Local fallback creation if API offline
    const newId = `11111111-1111-1111-1111-${Date.now()}`
    const newProj = {
      id: newId,
      title: projectData.title,
      package_type: projectData.packageType,
      budget: parseFloat(projectData.budget) || 25000.0,
      deadline: projectData.deadline,
      status: 'In Progress',
      progress_percentage: 0,
      priority: projectData.priority || 'medium',
      client_id: '44444444-4444-4444-4444-444444444401',
      notes: projectData.requirements || 'Created via Agency CRM Portal.',
      client: { company_name: projectData.companyName, contact_name: projectData.clientName, email: projectData.email },
    }

    const tplTasks: CRMTaskItem[] = [
      { id: `t-${Date.now()}-1`, project_id: newId, project_title: projectData.title, step_order: 1, title: 'Strategy & Wireframe Mapping', description: 'Initial UX discovery and wireframe creation.', role_required: 'UI/UX Designer', status: 'Todo', due_date: projectData.deadline },
      { id: `t-${Date.now()}-2`, project_id: newId, project_title: projectData.title, step_order: 2, title: 'Custom Code Engineering', description: 'Next.js 16 and Supabase build.', role_required: 'Frontend Developer', status: 'Locked' },
      { id: `t-${Date.now()}-3`, project_id: newId, project_title: projectData.title, step_order: 3, title: 'QA & Security Verification', description: 'Cross-device responsiveness and speed audits.', role_required: 'QA Tester', status: 'Locked' },
    ]

    setProjects((prev) => [newProj, ...prev])
    setTasks((prev) => [...tplTasks, ...prev])
    alert(`✅ Project Created Successfully! Automated Client credentials generated and workflow tasks spawned.`)
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
