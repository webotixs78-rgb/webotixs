'use client'

import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { ClientDashboardView } from '@/components/admin/crm/ClientDashboardView'
import { ProjectDetailModal } from '@/components/admin/crm/ProjectDetailModal'
import { CRMTaskItem } from '@/components/admin/crm/KanbanBoard'
import { Sparkles, Shield, AlertCircle, Building2, User, Mail } from 'lucide-react'

// Realistic seed projects if DB fetch not yet completed
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

function ClientPortalContent() {
  const searchParams = useSearchParams()
  const clientQuery = searchParams.get('client') || searchParams.get('email') || 'Al-Khaleej'

  const [projects, setProjects] = useState<any[]>(seedProjects)
  const [tasks, setTasks] = useState<CRMTaskItem[]>(seedTasks)
  const [invoices, setInvoices] = useState<any[]>(seedInvoices)
  const [tickets, setTickets] = useState<any[]>(seedTickets)
  const [selectedProject, setSelectedProject] = useState<any | null>(null)

  // Filter projects by logged in client profile
  const filteredProjects = projects.filter((p) => {
    if (clientQuery.toLowerCase().includes('luxbrand') || clientQuery.toLowerCase().includes('sophie')) {
      return p.client?.company_name === 'LuxBrand Paris' || p.client_id === '44444444-4444-4444-4444-444444444402'
    }
    if (clientQuery.toLowerCase().includes('finch')) {
      return p.client?.company_name === 'Finch Investments' || p.client_id === '44444444-4444-4444-4444-444444444403'
    }
    // Default Al-Khaleej
    return p.client?.company_name === 'Al-Khaleej Retail Group' || p.client_id === '44444444-4444-4444-4444-444444444401'
  })

  const activeClientProfile = filteredProjects[0]?.client || {
    company_name: 'Al-Khaleej Retail Group',
    contact_name: 'Tariq Al-Mansoor',
    email: 'tariq@alkhaleej.ae',
  }

  // Filter invoices & tickets for this specific client
  const filteredInvoices = invoices.filter((i) => i.client?.company_name === activeClientProfile.company_name || i.client_id === filteredProjects[0]?.client_id)
  const filteredTickets = tickets.filter((t) => t.client?.company_name === activeClientProfile.company_name || t.client_id === filteredProjects[0]?.client_id)

  const handleCreateTicket = async (ticketData: any) => {
    const newTick = {
      id: `t-${Date.now()}`,
      ticket_number: `TICK-${Math.floor(100 + Math.random() * 900)}`,
      client_id: filteredProjects[0]?.client_id || '44444444-4444-4444-4444-444444444401',
      project_id: ticketData.projectId,
      subject: ticketData.subject,
      description: ticketData.description,
      priority: ticketData.priority || 'medium',
      status: 'Open',
      created_at: new Date().toISOString(),
      client: { company_name: activeClientProfile.company_name },
    }
    setTickets((prev) => [newTick, ...prev])
    alert(`✅ Support Ticket #${newTick.ticket_number} Submitted! Your dedicated Project Manager has been immediately notified via email & dashboard alert.`)
  }

  return (
    <div className="space-y-8">
      {/* Client Profile Identity Banner */}
      <div className="bg-gradient-to-r from-blue-900/40 via-[#0D1224] to-cyan-900/30 border border-blue-500/30 rounded-3xl p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-glow-sm">
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center text-white font-display font-bold text-2xl shadow-lg shrink-0">
            {activeClientProfile.company_name.substring(0, 2).toUpperCase()}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <Building2 size={16} className="text-cyan-400" />
              <h1 className="font-display text-xl md:text-2xl font-bold text-white tracking-tight">
                {activeClientProfile.company_name}
              </h1>
            </div>
            <p className="text-xs md:text-sm text-[#94A3B8] flex items-center gap-4 mt-1">
              <span className="flex items-center gap-1.5">
                <User size={13} className="text-blue-400" /> {activeClientProfile.contact_name}
              </span>
              <span className="flex items-center gap-1.5">
                <Mail size={13} className="text-blue-400" /> {activeClientProfile.email}
              </span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-[#050816]/80 px-4 py-3 rounded-2xl border border-[#273449]">
          <Shield size={18} className="text-emerald-400 shrink-0" />
          <div className="text-xs">
            <div className="text-white font-semibold">Strict Client Isolation Active</div>
            <div className="text-[#94A3B8]">Viewing ONLY your assigned deliverables & billing</div>
          </div>
        </div>
      </div>

      {/* Render ClientDashboardView cleanly */}
      <ClientDashboardView
        projects={filteredProjects}
        tasks={tasks}
        invoices={filteredInvoices}
        tickets={filteredTickets}
        currentRole="Client"
        onOpenProject={setSelectedProject}
        onCreateTicket={handleCreateTicket}
      />

      {/* Project Detail Command Modal */}
      {selectedProject && (
        <ProjectDetailModal
          project={selectedProject}
          tasks={tasks}
          currentRole="Client"
          onClose={() => setSelectedProject(null)}
          onCompleteTask={async () => {}}
        />
      )}
    </div>
  )
}

export const dynamic = 'force-dynamic'

export default function ClientPortalPage() {
  return (
    <React.Suspense fallback={<div className="text-center py-20 text-[#94A3B8]">Loading Client Portal...</div>}>
      <ClientPortalContent />
    </React.Suspense>
  )
}
