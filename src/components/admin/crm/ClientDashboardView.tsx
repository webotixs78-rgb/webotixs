'use client'

import React, { useState } from 'react'
import { CheckCircle2, Clock, ExternalLink, Download, MessageSquare, LifeBuoy, FileText, Receipt, User, Shield, Sparkles, Plus, AlertCircle, ThumbsUp, X } from 'lucide-react'
import { cn, formatDate } from '@/lib/utils'
import { CRMTaskItem } from './KanbanBoard'
import { CRMRole } from './CRMRoleSwitcher'

interface ClientDashboardViewProps {
  projects: any[]
  tasks: CRMTaskItem[]
  invoices: any[]
  tickets: any[]
  currentRole: CRMRole
  onOpenProject: (project: any) => void
  onCreateTicket: (ticketData: any) => Promise<void>
}

export function ClientDashboardView({
  projects,
  tasks,
  invoices,
  tickets,
  currentRole,
  onOpenProject,
  onCreateTicket,
}: ClientDashboardViewProps) {
  const [activeTab, setActiveTab] = useState<'my-projects' | 'files' | 'invoices' | 'tickets' | 'support'>('my-projects')
  const [isTicketModalOpen, setIsTicketModalOpen] = useState(false)
  const [ticketSubject, setTicketSubject] = useState('')
  const [ticketDesc, setTicketDesc] = useState('')
  const [ticketPriority, setTicketPriority] = useState('medium')
  const [submitting, setSubmitting] = useState(false)

  const clientProjects = projects.length > 0 ? projects : [
    {
      id: '55555555-5555-5555-5555-555555555501',
      title: 'Al-Khaleej E-Commerce Headless Storefront',
      package_type: 'Enterprise E-Commerce',
      budget: 45000.0,
      deadline: '2026-09-15',
      status: 'In Progress',
      progress_percentage: 60,
      client: { company_name: 'Al-Khaleej Retail Group', contact_name: 'Tariq Al-Mansoor', email: 'tariq@alkhaleej.ae' },
    },
  ]

  const activeProject = clientProjects[0]
  const projectTasks = tasks.filter((t) => t.project_id === activeProject.id)
  const clientInvoices = invoices.filter((inv) => inv.client_id === activeProject.client_id || inv.project_id === activeProject.id)
  const clientTickets = tickets.filter((t) => t.client_id === activeProject.client_id || t.project_id === activeProject.id)

  const handleTicketSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await onCreateTicket({
        clientId: activeProject.client_id || '44444444-4444-4444-4444-444444444401',
        projectId: activeProject.id,
        subject: ticketSubject,
        description: ticketDesc,
        priority: ticketPriority,
      })
      setIsTicketModalOpen(false)
      setTicketSubject('')
      setTicketDesc('')
    } finally {
      setSubmitting(false)
    }
  }

  const progress = activeProject.progress_percentage ?? 0

  return (
    <div className="space-y-6">
      {/* Client Welcome Header */}
      <div className="bg-gradient-to-r from-indigo-950/60 via-blue-950/40 to-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-glow-sm">
            <Sparkles size={28} />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="font-display font-bold text-white text-2xl">Client Portal Dashboard</h2>
              <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold uppercase tracking-wider">
                VIP Access
              </span>
            </div>
            <p className="text-[#94A3B8] text-xs mt-1">
              Welcome back, <strong className="text-white">{activeProject.client?.contact_name || 'Tariq Al-Mansoor'}</strong> ({activeProject.client?.company_name || 'Al-Khaleej Retail Group'}). Monitor real-time stage transitions, download files, and review invoices.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsTicketModalOpen(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs rounded-xl shadow-glow-sm transition-all"
          >
            <Plus size={16} />
            Raise Support Ticket
          </button>
        </div>
      </div>

      {/* Nav Tabs */}
      <div className="flex items-center justify-between border-b border-[#273449] pb-3 overflow-x-auto gap-2">
        <div className="flex items-center gap-2">
          {[
            { id: 'my-projects', label: 'My Active Projects', icon: Sparkles, count: clientProjects.length },
            { id: 'files', label: 'Uploaded Deliverables', icon: FileText, count: projectTasks.filter((t) => t.deliverable_url).length },
            { id: 'invoices', label: 'Invoices & Billing', icon: Receipt, count: clientInvoices.length },
            { id: 'tickets', label: 'My Support Tickets', icon: LifeBuoy, count: clientTickets.length },
          ].map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-all whitespace-nowrap',
                  isActive
                    ? 'bg-indigo-600 text-white shadow-glow-sm'
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
      </div>

      {/* TAB 1: MY PROJECTS & PROGRESS */}
      {activeTab === 'my-projects' && (
        <div className="space-y-6">
          {clientProjects.map((proj) => {
            const projProgress = proj.progress_percentage ?? 0
            const pTasks = tasks.filter((t) => t.project_id === proj.id)
            const isCompleted = projProgress === 100

            return (
              <div key={proj.id} className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 md:p-8 shadow-2xl space-y-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#273449] pb-6">
                  <div>
                    <div className="flex items-center gap-2.5">
                      <span className="px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-bold uppercase tracking-wider">
                        {proj.package_type || 'Custom E-Commerce'}
                      </span>
                      <span className={cn(
                        'px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border',
                        isCompleted ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30'
                      )}>
                        {proj.status}
                      </span>
                    </div>
                    <h3 className="font-display font-bold text-white text-2xl mt-2">{proj.title}</h3>
                    <p className="text-[#94A3B8] text-xs mt-1">
                      Estimated Delivery Deadline: <span className="text-amber-400 font-mono font-bold">{proj.deadline}</span>
                    </p>
                  </div>

                  <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                    <button
                      onClick={() => onOpenProject(proj)}
                      className="px-5 py-3 rounded-xl bg-[#050816] border border-[#273449] hover:border-blue-500/50 text-white font-bold text-xs transition-all flex items-center justify-center gap-2"
                    >
                      <MessageSquare size={15} className="text-blue-400" />
                      Open Project Command Center & Chat
                    </button>

                    {isCompleted ? (
                      <button
                        onClick={() => alert(`🎉 Thank you! You have formally approved delivery for ${proj.title}.`)}
                        className="px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs shadow-glow-sm transition-all flex items-center justify-center gap-2"
                      >
                        <ThumbsUp size={16} />
                        Approve Final Delivery & Sign-off
                      </button>
                    ) : (
                      <div className="px-4 py-2.5 rounded-xl bg-blue-600/10 border border-blue-500/20 text-blue-300 text-xs font-semibold text-center">
                        ⏳ Agency Team Actively Executing Steps
                      </div>
                    )}
                  </div>
                </div>

                {/* Animated Progress Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm font-bold">
                    <span className="text-white">Real-Time Stage Completion Velocity</span>
                    <span className="text-blue-400 font-display text-lg">{projProgress}% Delivered</span>
                  </div>
                  <div className="w-full h-3.5 rounded-full bg-[#050816] border border-[#273449] overflow-hidden p-0.5">
                    <div
                      className="h-full bg-gradient-to-r from-blue-600 via-indigo-500 to-emerald-400 rounded-full transition-all duration-1000 shadow-glow-sm"
                      style={{ width: `${projProgress}%` }}
                    />
                  </div>
                </div>

                {/* Sequential Stage Timeline Grid */}
                <div className="space-y-3 pt-2">
                  <h4 className="text-xs font-bold text-[#94A3B8] uppercase tracking-wider">Lifecycle Workflow Steps ({pTasks.length})</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
                    {pTasks.map((t, idx) => (
                      <div
                        key={t.id}
                        onClick={() => onOpenProject(proj)}
                        className={cn(
                          'p-4 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between space-y-2',
                          t.status === 'Completed'
                            ? 'bg-emerald-500/10 border-emerald-500/40 hover:border-emerald-500'
                            : t.status === 'Locked'
                            ? 'bg-[#050816]/60 border-[#273449]/40 opacity-60'
                            : 'bg-blue-600/10 border-blue-500/50 shadow-md ring-1 ring-blue-500/20'
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">Step #{t.step_order}</span>
                          {t.status === 'Completed' ? <CheckCircle2 size={16} className="text-emerald-400" /> : <Clock size={16} className="text-blue-400" />}
                        </div>

                        <h5 className="font-bold text-white text-xs leading-tight line-clamp-2">{t.title}</h5>

                        <div className="pt-2 border-t border-[#273449]/50 flex items-center justify-between text-[10px] font-semibold">
                          <span className="text-[#94A3B8] truncate">{t.role_required}</span>
                          <span className={cn(
                            t.status === 'Completed' ? 'text-emerald-400 font-bold' : t.status === 'Locked' ? 'text-slate-400' : 'text-blue-400 font-bold'
                          )}>
                            {t.status}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* TAB 2: FILES & DELIVERABLES */}
      {activeTab === 'files' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#273449] pb-4">
            <div>
              <h3 className="font-display font-bold text-white text-lg">Verified Deliverables & Files</h3>
              <p className="text-[#94A3B8] text-xs">Download or inspect Figma designs, staging URLs, and brand documentation.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {tasks
              .filter((t) => t.deliverable_url)
              .map((t) => (
                <div key={t.id} className="p-4 rounded-2xl bg-[#050816] border border-[#273449] hover:border-indigo-500/50 transition-all space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Step #{t.step_order} Deliverable</span>
                    <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[9px] font-bold">
                      Ready
                    </span>
                  </div>
                  <h4 className="font-bold text-white text-sm truncate">{t.title}</h4>
                  <a
                    href={t.deliverable_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-3 rounded-xl bg-[#0D1224] border border-[#273449] text-blue-400 text-xs hover:underline flex items-center justify-between gap-2 truncate font-mono"
                  >
                    <span className="truncate">{t.deliverable_url}</span>
                    <ExternalLink size={14} className="shrink-0" />
                  </a>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* TAB 3: INVOICES & PDF DOWNLOADS */}
      {activeTab === 'invoices' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">My Invoices & Billing Statements</h3>
            <p className="text-[#94A3B8] text-xs">Download PDF invoice receipts or verify milestone payment schedules.</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#273449] text-[#94A3B8] uppercase tracking-wider text-left">
                  <th className="py-3 px-4 font-semibold">Invoice #</th>
                  <th className="py-3 px-4 font-semibold">Description</th>
                  <th className="py-3 px-4 font-semibold">Total Amount</th>
                  <th className="py-3 px-4 font-semibold">Issue Date</th>
                  <th className="py-3 px-4 font-semibold text-center">Status</th>
                  <th className="py-3 px-4 font-semibold text-right">Download Receipt</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((inv) => (
                  <tr key={inv.id} className="border-b border-[#273449]/60 hover:bg-white/[0.02]">
                    <td className="py-3.5 px-4 font-mono font-bold text-white">{inv.invoice_number}</td>
                    <td className="py-3.5 px-4 text-[#94A3B8]">{inv.notes || '50% Milestone Deposit for E-Commerce Overhaul'}</td>
                    <td className="py-3.5 px-4 font-bold text-emerald-400">${inv.total_amount?.toLocaleString() || '23,625.00'}</td>
                    <td className="py-3.5 px-4 text-[#94A3B8] font-mono">{inv.issue_date}</td>
                    <td className="py-3.5 px-4 text-center">
                      <span className={cn('px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase border', inv.status === 'Paid' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30')}>
                        {inv.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => alert(`📥 Downloading PDF Invoice Receipt for ${inv.invoice_number}...`)}
                        className="px-3 py-1.5 rounded-xl bg-[#050816] border border-[#273449] hover:border-blue-500 text-blue-400 font-bold inline-flex items-center gap-1.5 transition-colors"
                      >
                        <Download size={13} /> Download PDF
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 4: SUPPORT TICKETS */}
      {activeTab === 'tickets' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-display font-bold text-white text-lg">My Support Tickets</h3>
              <p className="text-[#94A3B8] text-xs">Direct high-priority communication pipeline with your Project Manager.</p>
            </div>
            <button
              onClick={() => setIsTicketModalOpen(true)}
              className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs flex items-center gap-1.5"
            >
              <Plus size={14} /> New Ticket
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {tickets.map((t) => (
              <div key={t.id} className="p-4 rounded-2xl bg-[#050816] border border-[#273449] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-xs text-blue-400">{t.ticket_number} • {t.priority} Priority</span>
                  <span className={cn('px-2 py-0.5 rounded text-[10px] font-bold uppercase border', t.status === 'Solved' || t.status === 'Closed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30')}>
                    {t.status}
                  </span>
                </div>
                <h4 className="font-bold text-white text-sm">{t.subject}</h4>
                <p className="text-[#94A3B8] text-xs leading-relaxed">{t.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* NEW TICKET MODAL */}
      {isTicketModalOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="w-full max-w-lg bg-[#0D1224] border border-[#273449] rounded-3xl p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-[#273449] pb-4">
              <h3 className="font-display font-bold text-white text-lg flex items-center gap-2">
                <LifeBuoy size={18} className="text-indigo-400" /> Open New Support Ticket
              </h3>
              <button onClick={() => setIsTicketModalOpen(false)} className="p-2 rounded-xl hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleTicketSubmit} className="space-y-4">
              <div className="space-y-1">
                <label className="text-xs font-bold text-white">Subject *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Inquiry regarding staging checkout currency persistence"
                  value={ticketSubject}
                  onChange={(e) => setTicketSubject(e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-indigo-500/50"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-white">Priority</label>
                <select
                  value={ticketPriority}
                  onChange={(e) => setTicketPriority(e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-indigo-500/50"
                >
                  <option value="low">Low Priority</option>
                  <option value="medium">Medium Priority</option>
                  <option value="high">High Priority</option>
                  <option value="urgent">Urgent / Blocker</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-white">Description & Specific Details *</label>
                <textarea
                  rows={4}
                  required
                  placeholder="Provide step-by-step details or questions for your Project Manager..."
                  value={ticketDesc}
                  onChange={(e) => setTicketDesc(e.target.value)}
                  className="w-full px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-indigo-500/50"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsTicketModalOpen(false)}
                  className="px-5 py-2.5 rounded-xl bg-[#050816] border border-[#273449] text-xs font-bold text-[#94A3B8] hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white font-bold text-xs shadow-glow-sm transition-all"
                >
                  {submitting ? 'Opening Ticket...' : 'Submit Support Ticket'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
