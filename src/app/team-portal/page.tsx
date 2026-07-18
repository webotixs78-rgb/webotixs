'use client'

import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { TeamDashboardView } from '@/components/admin/crm/TeamDashboardView'
import { ProjectDetailModal } from '@/components/admin/crm/ProjectDetailModal'
import { CRMTaskItem } from '@/components/admin/crm/KanbanBoard'
import { CRMRole } from '@/components/admin/crm/CRMRoleSwitcher'
import { Sparkles, Shield, AlertCircle, Briefcase, User, CheckCircle2 } from 'lucide-react'

// Seed data fallback
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
]

function TeamPortalContent() {
  const searchParams = useSearchParams()
  const roleParam = (searchParams.get('role') as CRMRole) || 'UI/UX Designer'
  const emailParam = searchParams.get('email') || 'sarah@webotixs.com'

  const [projects, setProjects] = useState<any[]>(seedProjects)
  const [tasks, setTasks] = useState<CRMTaskItem[]>(seedTasks)
  const [selectedProject, setSelectedProject] = useState<any | null>(null)

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
    alert(`✅ Task Complete & Deliverable Submitted! Next pipeline task automatically unlocked.`)
  }

  const handleTaskStatusChange = (taskId: string, newStatus: CRMTaskItem['status']) => {
    setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t)))
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
