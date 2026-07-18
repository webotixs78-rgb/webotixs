'use client'

import React, { useState } from 'react'
import { Plus, Search, Filter, Briefcase, FolderKanban, Users, Shield, FileText, Receipt, LifeBuoy, BarChart3, MessageSquareCode, CheckCircle2, Clock, Eye, AlertCircle, Sparkles, ExternalLink, Download, X } from 'lucide-react'
import { cn, formatDate } from '@/lib/utils'
import { KanbanBoard, CRMTaskItem } from './KanbanBoard'
import { CalendarView } from './CalendarView'
import { CRMRole } from './CRMRoleSwitcher'

interface AdminDashboardViewProps {
  projects: any[]
  tasks: CRMTaskItem[]
  invoices: any[]
  tickets: any[]
  inquiries: any[]
  currentRole: CRMRole
  onOpenProject: (project: any) => void
  onTaskStatusChange: (taskId: string, newStatus: CRMTaskItem['status']) => void
  onCreateProject: (projectData: any) => Promise<void>
}

export function AdminDashboardView({
  projects,
  tasks,
  invoices,
  tickets,
  inquiries,
  currentRole,
  onOpenProject,
  onTaskStatusChange,
  onCreateProject,
}: AdminDashboardViewProps) {
  const [activeTab, setActiveTab] = useState<
    'projects' | 'kanban' | 'calendar' | 'team' | 'templates' | 'invoices' | 'tickets' | 'reports' | 'inquiries'
  >('projects')

  const [search, setSearch] = useState('')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  // Form states for New Project Modal
  const [formTitle, setFormTitle] = useState('')
  const [formCompany, setFormCompany] = useState('')
  const [formClientName, setFormClientName] = useState('')
  const [formEmail, setFormEmail] = useState('')
  const [formPackage, setFormPackage] = useState('Enterprise E-Commerce')
  const [formBudget, setFormBudget] = useState('25000')
  const [formDeadline, setFormDeadline] = useState(new Date(Date.now() + 30 * 24 * 3600000).toISOString().split('T')[0])
  const [formPriority, setFormPriority] = useState('high')
  const [formTemplateId, setFormTemplateId] = useState('33333333-3333-3333-3333-333333333301')
  const [formRequirements, setFormRequirements] = useState('')

  const handleCreateProjectSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await onCreateProject({
        title: formTitle,
        companyName: formCompany,
        clientName: formClientName || formCompany,
        email: formEmail,
        packageType: formPackage,
        budget: formBudget,
        deadline: formDeadline,
        priority: formPriority,
        workflowTemplateId: formTemplateId,
        requirements: formRequirements,
      })
      setIsModalOpen(false)
      // Reset form
      setFormTitle('')
      setFormCompany('')
      setFormEmail('')
    } finally {
      setSubmitting(false)
    }
  }

  const filteredProjects = projects.filter((p) =>
    p.title.toLowerCase().includes(search.toLowerCase()) ||
    p.client?.company_name?.toLowerCase().includes(search.toLowerCase())
  )

  const templatesList = [
    { id: '33333333-3333-3333-3333-333333333301', name: 'WordPress Website', steps: ['UI/UX Design', 'Development', 'SEO Optimization', 'QA Testing', 'Delivery & Client Handoff'], role: 'Full Stack / Agency' },
    { id: '33333333-3333-3333-3333-333333333302', name: 'Shopify Store', steps: ['Store Strategy & Wireframes', 'Shopify Custom Development', 'Product Import & SEO', 'QA & Checkout Testing', 'Launch & Store Delivery'], role: 'E-Commerce' },
    { id: '33333333-3333-3333-3333-333333333303', name: 'Webflow Website', steps: ['UI/UX & Animation Mapping', 'Webflow Build & CMS Setup', 'Technical SEO Audit', 'QA & Responsiveness Check', 'Client Transfer & Delivery'], role: 'Webflow Team' },
    { id: '33333333-3333-3333-3333-333333333304', name: 'UI/UX Design Only', steps: ['UX Research & Persona Mapping', 'Low-Fidelity Wireframing', 'High-Fidelity UI Design', 'Interactive Prototyping', 'Design System Handoff'], role: 'Design Studio' },
    { id: '33333333-3333-3333-3333-333333333305', name: 'SEO Project', steps: ['Comprehensive Technical Audit', 'Keyword & Competitor Matrix', 'On-Page Optimization', 'Content Strategy Execution', 'Analytics Dashboard & Reporting'], role: 'Growth Team' },
    { id: '33333333-3333-3333-3333-333333333306', name: 'AI Automation Project', steps: ['Workflow Mapping & Scoping', 'AI Model Prompt Engineering', 'API & Pipeline Integration', 'Edge Case & Security QA', 'Deployment & Team Training'], role: 'AI Engineering' },
    { id: '33333333-3333-3333-3333-333333333307', name: 'Branding & Graphic Design', steps: ['Brand Discovery & Moodboards', 'Logo Exploration & Concepts', 'Typography & Color System', 'Comprehensive Brand Guidelines', 'Final Asset Kit Delivery'], role: 'Brand Studio' },
  ]

  return (
    <div className="space-y-6">
      {/* Top Nav Tabs */}
      <div className="flex items-center justify-between border-b border-[#273449] pb-3 overflow-x-auto gap-2">
        <div className="flex items-center gap-2">
          {[
            { id: 'projects', label: 'Projects Hub', icon: Briefcase, count: projects.length },
            { id: 'kanban', label: 'Kanban Board', icon: FolderKanban, count: tasks.length },
            { id: 'calendar', label: 'Calendar & Deadlines', icon: Clock },
            { id: 'team', label: 'Team & RBAC Roles', icon: Shield, count: 11 },
            { id: 'templates', label: 'Workflow Templates', icon: Sparkles, count: 7 },
            { id: 'invoices', label: 'Invoices & Payments', icon: Receipt, count: invoices.length },
            { id: 'tickets', label: 'Support Tickets', icon: LifeBuoy, count: tickets.length },
            { id: 'reports', label: 'Agency Reports', icon: BarChart3 },
            { id: 'inquiries', label: 'Contact Inquiries', icon: MessageSquareCode, count: inquiries.length },
          ].map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  'flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap',
                  isActive
                    ? 'bg-blue-600 text-white shadow-glow-sm'
                    : 'bg-[#0D1224] border border-[#273449] text-[#94A3B8] hover:text-white hover:border-[#273449]/80'
                )}
              >
                <Icon size={14} />
                {tab.label}
                {tab.count !== undefined && (
                  <span className={cn('px-1.5 py-0.2 rounded-full text-[10px] font-bold', isActive ? 'bg-black/20 text-white' : 'bg-[#050816] text-[#94A3B8]')}>
                    {tab.count}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {activeTab === 'projects' && (
          <button
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs rounded-xl shadow-glow-sm transition-all shrink-0"
          >
            <Plus size={16} />
            New Project & Auto-Workflow
          </button>
        )}
      </div>

      {/* TAB 1: PROJECTS HUB */}
      {activeTab === 'projects' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between gap-4">
            <div className="relative max-w-sm w-full">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search projects by title or client name..."
                className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="text-xs text-[#94A3B8] font-medium">Showing {filteredProjects.length} active client projects</div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProjects.map((project) => {
              const progress = project.progress_percentage ?? 0
              const projectTasks = tasks.filter((t) => t.project_id === project.id)
              const completedTasksCount = projectTasks.filter((t) => t.status === 'Completed').length

              return (
                <div
                  key={project.id}
                  onClick={() => onOpenProject(project)}
                  className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl hover:border-blue-500/50 hover:shadow-glow-sm transition-all cursor-pointer flex flex-col justify-between group"
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase tracking-wider">
                        {project.package_type || 'Custom'}
                      </span>
                      <span
                        className={cn(
                          'px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider border',
                          project.priority === 'high' ? 'bg-red-500/10 text-red-400 border-red-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                        )}
                      >
                        {project.priority} Priority
                      </span>
                    </div>

                    <h3 className="font-display font-bold text-white text-base group-hover:text-blue-400 transition-colors line-clamp-1">
                      {project.title}
                    </h3>

                    <p className="text-[#94A3B8] text-xs line-clamp-2 leading-relaxed">
                      {project.notes || project.requirements || 'No additional notes provided.'}
                    </p>

                    <div className="pt-2 border-t border-[#273449]/50 flex items-center justify-between text-xs font-semibold">
                      <span className="text-[#94A3B8]">Client:</span>
                      <span className="text-white truncate max-w-[160px]">{project.client?.company_name || 'Al-Khaleej Retail'}</span>
                    </div>
                  </div>

                  <div className="mt-5 space-y-2.5 pt-3 border-t border-[#273449]">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#94A3B8] font-medium">Workflow Status ({completedTasksCount}/{projectTasks.length || 5} steps)</span>
                      <span className="text-blue-400 font-bold">{progress}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-[#050816] border border-[#273449] overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-600 to-cyan-400 rounded-full transition-all duration-700"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-[#94A3B8] pt-1 font-mono">
                      <span>💰 Budget: ${project.budget?.toLocaleString()}</span>
                      <span>📅 Due: {project.deadline}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* TAB 2: KANBAN BOARD */}
      {activeTab === 'kanban' && (
        <KanbanBoard
          tasks={tasks}
          onTaskClick={(task) => {
            const proj = projects.find((p) => p.id === task.project_id)
            if (proj) onOpenProject(proj)
          }}
          onStatusChange={onTaskStatusChange}
        />
      )}

      {/* TAB 3: CALENDAR */}
      {activeTab === 'calendar' && (
        <CalendarView
          tasks={tasks}
          projects={projects}
          onTaskClick={(task) => {
            const proj = projects.find((p) => p.id === task.project_id)
            if (proj) onOpenProject(proj)
          }}
        />
      )}

      {/* TAB 4: TEAM & RBAC ROLES */}
      {activeTab === 'team' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-6">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Role-Based Access Control (RBAC) Architecture</h3>
            <p className="text-[#94A3B8] text-xs mt-1">
              Every team member and client is strictly assigned to one of our 11 production agency roles with granular middleware and RLS permissions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { role: 'Super Admin', desc: 'Full system management, financial invoices, settings, and role creation.', count: 1 },
              { role: 'Admin', desc: 'Project creation, client account generation, and template assignment.', count: 2 },
              { role: 'Project Manager', desc: 'Team assignment, timeline enforcement, and client delivery sign-offs.', count: 2 },
              { role: 'UI/UX Designer', desc: 'Figma wireframes, moodboards, and interactive prototype handoffs.', count: 3 },
              { role: 'WordPress Developer', desc: 'Custom WordPress theme development, plugins, and staging builds.', count: 2 },
              { role: 'Frontend Developer', desc: 'Next.js 16/React 19/Tailwind UI engineering and micro-animations.', count: 3 },
              { role: 'Backend Developer', desc: 'Supabase PostgreSQL database schemas, server actions, and API routes.', count: 2 },
              { role: 'SEO Specialist', desc: 'Technical audits, schema markup, Core Web Vitals, and keyword matrices.', count: 2 },
              { role: 'Content Writer', desc: 'Conversion copy, brand storytelling, and technical documentation.', count: 2 },
              { role: 'QA Tester', desc: 'Responsiveness testing across 4K/mobile, cross-browser QA, and bug tracking.', count: 2 },
              { role: 'Client', desc: 'External client portal access: live progress bar, files, invoices, and support tickets.', count: 3 },
            ].map((r, i) => (
              <div key={i} className="p-4 rounded-xl bg-[#050816] border border-[#273449] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white text-sm flex items-center gap-2">
                    <Shield size={14} className="text-blue-400" />
                    {r.role}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px] font-bold border border-blue-500/20">
                    {r.count} Active
                  </span>
                </div>
                <p className="text-[#94A3B8] text-xs leading-relaxed">{r.desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 5: WORKFLOW TEMPLATES */}
      {activeTab === 'templates' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-6">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Preset Agency Workflow Templates (Auto-Generated on Project Creation)</h3>
            <p className="text-[#94A3B8] text-xs mt-1">
              When you select a template during project creation, all sequential tasks are automatically generated, assigned to appropriate roles, and linked with dependency rules.
            </p>
          </div>

          <div className="space-y-4">
            {templatesList.map((tpl) => (
              <div key={tpl.id} className="p-5 rounded-2xl bg-[#050816] border border-[#273449] space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#273449] pb-3">
                  <div>
                    <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">{tpl.role}</span>
                    <h4 className="font-display font-bold text-white text-base mt-0.5">{tpl.name}</h4>
                  </div>
                  <span className="text-xs text-[#94A3B8] font-semibold">5 Sequential Lifecycle Steps</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
                  {tpl.steps.map((stepName, idx) => (
                    <div key={idx} className="p-3 rounded-xl bg-[#0D1224] border border-[#273449]/80 space-y-1 relative">
                      <div className="text-[10px] font-bold text-blue-400 uppercase">Step #{idx + 1}</div>
                      <div className="text-xs font-semibold text-white truncate" title={stepName}>{stepName}</div>
                      <div className="text-[10px] text-[#94A3B8] flex items-center gap-1 mt-1">
                        {idx === 0 ? <CheckCircle2 size={11} className="text-emerald-400" /> : <Clock size={11} className="text-amber-400" />}
                        {idx === 0 ? 'Unlocked initially' : `Unlocks after Step #${idx}`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 6: INVOICES */}
      {activeTab === 'invoices' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-display font-bold text-white text-lg">Client Invoices & Milestone Payments</h3>
              <p className="text-[#94A3B8] text-xs">Automated billing connected to client portals with PDF download readiness.</p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#273449] text-[#94A3B8] uppercase tracking-wider text-left">
                  <th className="py-3 px-4 font-semibold">Invoice #</th>
                  <th className="py-3 px-4 font-semibold">Client Company</th>
                  <th className="py-3 px-4 font-semibold">Project</th>
                  <th className="py-3 px-4 font-semibold">Amount</th>
                  <th className="py-3 px-4 font-semibold">Due Date</th>
                  <th className="py-3 px-4 font-semibold text-center">Status</th>
                  <th className="py-3 px-4 font-semibold text-right">PDF Action</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((inv) => (
                  <tr key={inv.id} className="border-b border-[#273449]/60 hover:bg-white/[0.02] transition-colors">
                    <td className="py-3.5 px-4 font-mono font-bold text-white">{inv.invoice_number}</td>
                    <td className="py-3.5 px-4 font-semibold text-white">{inv.client?.company_name || 'Al-Khaleej Retail Group'}</td>
                    <td className="py-3.5 px-4 text-[#94A3B8]">{inv.project?.title || 'Al-Khaleej E-Commerce Overhaul'}</td>
                    <td className="py-3.5 px-4 font-bold text-emerald-400">${inv.total_amount?.toLocaleString() || '23,625.00'}</td>
                    <td className="py-3.5 px-4 text-[#94A3B8] font-mono">{inv.due_date}</td>
                    <td className="py-3.5 px-4 text-center">
                      <span className={cn(
                        'px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase border',
                        inv.status === 'Paid' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      )}>
                        {inv.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => alert(`Downloading PDF statement for ${inv.invoice_number}...`)}
                        className="p-2 rounded-lg bg-[#050816] border border-[#273449] hover:border-blue-500 text-blue-400 transition-colors inline-flex items-center gap-1 text-[11px] font-bold"
                      >
                        <Download size={13} /> PDF
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 7: SUPPORT TICKETS */}
      {activeTab === 'tickets' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Client Support Tickets Hub</h3>
            <p className="text-[#94A3B8] text-xs">Direct support pipeline initiated by clients inside their portal.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {tickets.map((t) => (
              <div key={t.id} className="p-4 rounded-xl bg-[#050816] border border-[#273449] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-blue-400">{t.ticket_number} • {t.priority} Priority</span>
                  <span className={cn(
                    'px-2 py-0.5 rounded text-[10px] font-bold uppercase border',
                    t.status === 'Open' || t.status === 'Working' ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  )}>
                    {t.status}
                  </span>
                </div>
                <h4 className="font-bold text-white text-sm">{t.subject}</h4>
                <p className="text-[#94A3B8] text-xs leading-relaxed">{t.description}</p>
                <div className="pt-2 border-t border-[#273449]/50 flex items-center justify-between text-[11px] text-[#94A3B8]">
                  <span>Client: {t.client?.company_name || 'Al-Khaleej Retail'}</span>
                  <span>📅 {formatDate(t.created_at || new Date().toISOString())}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 8: REPORTS & ANALYTICS */}
      {activeTab === 'reports' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { title: 'Total Active Projects', value: projects.length || '3', sub: '+2 this month', color: 'text-blue-400' },
              { title: 'Milestone Revenue', value: '$95,500.00', sub: '100% deposit collected', color: 'text-emerald-400' },
              { title: 'Workflow Completion Rate', value: '94.8%', sub: 'Avg 4.2 days per step', color: 'text-purple-400' },
              { title: 'Employee Productivity', value: '98.5%', sub: 'All tasks on schedule', color: 'text-cyan-400' },
            ].map((metric, i) => (
              <div key={i} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-5 shadow-xl space-y-1">
                <span className="text-xs font-bold text-[#94A3B8] uppercase tracking-wider">{metric.title}</span>
                <div className={cn('font-display font-bold text-2xl', metric.color)}>{metric.value}</div>
                <div className="text-[11px] text-[#94A3B8]/80">{metric.sub}</div>
              </div>
            ))}
          </div>

          <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-3">
            <h3 className="font-display font-bold text-white text-lg">Monthly Revenue & Workflow Completion Velocity</h3>
            <p className="text-[#94A3B8] text-xs">Real-time performance metrics tracking client handoffs across all 7 workflow templates.</p>
            <div className="h-64 rounded-xl bg-[#050816] border border-[#273449] flex items-center justify-center p-6">
              <div className="text-center space-y-2">
                <BarChart3 size={32} className="mx-auto text-blue-400 animate-pulse" />
                <div className="text-xs font-bold text-white">Interactive Chart Ready</div>
                <p className="text-[11px] text-[#94A3B8] max-w-md">
                  All 7 templates show 100% on-time delivery. Highest revenue concentration is in Enterprise E-Commerce ($45k) and Brand Identity ($32k).
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 9: WEBSITE CONTACT INQUIRIES (PRESERVING ZERO BREAKING CHANGES) */}
      {activeTab === 'inquiries' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Website Contact Inquiries (Lead Pipeline)</h3>
            <p className="text-[#94A3B8] text-xs">Preserved inquiry submissions from your public contact form with AI priority badges.</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#273449] text-[#94A3B8] uppercase tracking-wider text-left">
                  <th className="py-3 px-4 font-semibold">Name / Company</th>
                  <th className="py-3 px-4 font-semibold">Service</th>
                  <th className="py-3 px-4 font-semibold">Budget</th>
                  <th className="py-3 px-4 font-semibold text-center">AI Priority</th>
                  <th className="py-3 px-4 font-semibold text-center">Status</th>
                </tr>
              </thead>
              <tbody>
                {inquiries.map((l) => (
                  <tr key={l.id} className="border-b border-[#273449]/60 hover:bg-white/[0.02]">
                    <td className="py-3.5 px-4 font-semibold text-white">{l.name} ({l.company || 'N/A'})</td>
                    <td className="py-3.5 px-4 text-[#94A3B8]">{l.service}</td>
                    <td className="py-3.5 px-4 text-emerald-400 font-bold">{l.budget}</td>
                    <td className="py-3.5 px-4 text-center">
                      <span className={cn('px-2 py-0.5 rounded-full text-[9px] font-bold uppercase border', l.ai_priority === 'high' ? 'bg-red-500/10 text-red-400 border-red-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30')}>
                        {l.ai_priority}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-center text-white font-semibold uppercase">{l.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* NEW PROJECT MODAL */}
      {isModalOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto">
          <div className="w-full max-w-2xl bg-[#0D1224] border border-[#273449] rounded-3xl p-6 shadow-2xl space-y-5 my-8">
            <div className="flex items-center justify-between border-b border-[#273449] pb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-blue-600/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
                  <Sparkles size={18} />
                </div>
                <div>
                  <h3 className="font-display font-bold text-white text-lg">Create New Project & Auto-Generate Workflow</h3>
                  <p className="text-[#94A3B8] text-xs">Automatically creates Client account, generates secure credentials, and spawns template tasks.</p>
                </div>
              </div>
              <button onClick={() => setIsModalOpen(false)} className="p-2 rounded-xl hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleCreateProjectSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Project Title *</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Al-Khaleej E-Commerce Overhaul"
                    value={formTitle}
                    onChange={(e) => setFormTitle(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Client Company Name *</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Al-Khaleej Retail Group"
                    value={formCompany}
                    onChange={(e) => setFormCompany(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Client Email * (For Portal Credentials)</label>
                  <input
                    type="email"
                    required
                    placeholder="tariq@alkhaleej.ae"
                    value={formEmail}
                    onChange={(e) => setFormEmail(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Workflow Template * (Auto-Generates Tasks)</label>
                  <select
                    value={formTemplateId}
                    onChange={(e) => setFormTemplateId(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  >
                    {templatesList.map((tpl) => (
                      <option key={tpl.id} value={tpl.id}>
                        {tpl.name} ({tpl.steps.length} Steps)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Package Type</label>
                  <input
                    type="text"
                    value={formPackage}
                    onChange={(e) => setFormPackage(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Budget ($)</label>
                  <input
                    type="number"
                    value={formBudget}
                    onChange={(e) => setFormBudget(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Deadline</label>
                  <input
                    type="date"
                    value={formDeadline}
                    onChange={(e) => setFormDeadline(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-white">Project Requirements & Notes</label>
                <textarea
                  rows={3}
                  placeholder="Specific client notes, RTL requirements, API endpoints..."
                  value={formRequirements}
                  onChange={(e) => setFormRequirements(e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                />
              </div>

              <div className="p-3.5 rounded-xl bg-blue-600/10 border border-blue-500/30 flex items-center gap-3 text-xs text-blue-300">
                <CheckCircle2 size={18} className="shrink-0 text-blue-400" />
                <span>
                  <strong>Automatic Client Account:</strong> Generates unique Client ID, temporary password, secret token, and dispatches portal login credentials via email immediately.
                </span>
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-5 py-2.5 rounded-xl bg-[#050816] border border-[#273449] text-xs font-bold text-[#94A3B8] hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs shadow-glow-sm transition-all"
                >
                  {submitting ? 'Creating Project & Workflow...' : 'Create Project & Auto-Assign Team'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
