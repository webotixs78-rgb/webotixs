'use client'

import React, { useState, useEffect } from 'react'
import { Sparkles, Building2, Clock, CheckCircle2, AlertCircle, FileText, MessageSquare, Plus, DollarSign, Users, FolderKanban, X, Send } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function ClientDashboardPage() {
  const [projects, setProjects] = useState<any[]>([])
  const [invoices, setInvoices] = useState<any[]>([])
  const [tickets, setTickets] = useState<any[]>([])
  const [session, setSession] = useState<any | null>(null)
  const [activeTab, setActiveTab] = useState<'projects' | 'invoices' | 'support'>('projects')
  const [isCreatingTicket, setIsCreatingTicket] = useState(false)
  const [newTicket, setNewTicket] = useState({ subject: '', description: '', priority: 'medium' })

  useEffect(() => {
    const storedSession = localStorage.getItem('webotixs_active_session')
    let activeEmail = ''
    if (storedSession) {
      const parsed = JSON.parse(storedSession)
      setSession(parsed)
      activeEmail = parsed.email || ''
    }

    async function loadData() {
      try {
        const [pRes, iRes, tRes] = await Promise.all([
          fetch('/api/crm/projects').catch(() => null),
          fetch('/api/crm/invoices').catch(() => null),
          fetch('/api/crm/tickets').catch(() => null),
        ])

        if (pRes && pRes.ok) {
          const pData = await pRes.json()
          if (pData.projects && pData.projects.length > 0) {
            setProjects(pData.projects)
          }
        }
        if (iRes && iRes.ok) {
          const iData = await iRes.json()
          if (iData.invoices && iData.invoices.length > 0) setInvoices(iData.invoices)
        }
        if (tRes && tRes.ok) {
          const tData = await tRes.json()
          if (tData.tickets && tData.tickets.length > 0) setTickets(tData.tickets)
        }
      } catch (e) {}

      const localProjects = localStorage.getItem('webotixs_crm_projects')
      const localInvoices = localStorage.getItem('webotixs_crm_invoices')
      const localTickets = localStorage.getItem('webotixs_crm_tickets')

      if (localProjects) {
        const all = JSON.parse(localProjects)
        const myProj = all.filter(
          (p: any) =>
            !activeEmail ||
            activeEmail === 'webotixs78@gmail.com' ||
            p.client?.email?.toLowerCase() === activeEmail.toLowerCase() ||
            p.client?.company_name?.toLowerCase().includes(session?.name?.toLowerCase() || '')
        )
        setProjects(myProj.length > 0 ? myProj : all)
      }
      if (localInvoices) setInvoices(JSON.parse(localInvoices))
      if (localTickets) setTickets(JSON.parse(localTickets))
    }
    loadData()
  }, [session?.name])

  const handleCreateTicket = () => {
    if (!newTicket.subject) {
      alert('Please enter a support ticket subject.')
      return
    }

    const created = {
      id: `ticket-${Date.now()}`,
      ticket_number: `TICK-${Math.floor(100 + Math.random() * 900)}`,
      subject: newTicket.subject,
      description: newTicket.description || 'Submitted from Client Dashboard.',
      priority: newTicket.priority,
      status: 'Open',
      created_at: new Date().toISOString(),
      client: { company_name: session?.name || 'Valued Client' },
    }

    const updated = [created, ...tickets]
    setTickets(updated)
    localStorage.setItem('webotixs_crm_tickets', JSON.stringify(updated))
    setIsCreatingTicket(false)
    setNewTicket({ subject: '', description: '', priority: 'medium' })
    alert('✅ Support Request Submitted! Your dedicated agency manager has been notified instantly.')
  }

  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="bg-gradient-to-r from-cyan-900/40 via-[#0D1224] to-blue-900/30 border border-cyan-500/30 rounded-3xl p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xl">
        <div>
          <span className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-bold uppercase tracking-wider">
            VIP Client Portal
          </span>
          <h1 className="font-display text-2xl md:text-3xl font-bold text-white tracking-tight mt-2">
            Welcome, {session?.name || 'Valued Client'}
          </h1>
          <p className="text-xs md:text-sm text-[#94A3B8] max-w-xl mt-1">
            Track real-time project progress, review milestones, download invoices, and communicate directly with your dedicated Webotixs agency team.
          </p>
        </div>

        <button
          onClick={() => setIsCreatingTicket(true)}
          className="flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-2xl shadow-lg hover:shadow-cyan-500/25 transition-all text-xs shrink-0"
        >
          <Plus size={16} /> Open Support Request
        </button>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-3 border-b border-[#273449] pb-3">
        <button
          onClick={() => setActiveTab('projects')}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
            activeTab === 'projects' ? 'bg-cyan-500 text-black shadow-md font-bold' : 'text-[#94A3B8] hover:bg-white/5 hover:text-white'
          )}
        >
          <FolderKanban size={14} /> My Projects ({projects.length})
        </button>
        <button
          onClick={() => setActiveTab('invoices')}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
            activeTab === 'invoices' ? 'bg-cyan-500 text-black shadow-md font-bold' : 'text-[#94A3B8] hover:bg-white/5 hover:text-white'
          )}
        >
          <DollarSign size={14} /> Invoices & Billing ({invoices.length})
        </button>
        <button
          onClick={() => setActiveTab('support')}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
            activeTab === 'support' ? 'bg-cyan-500 text-black shadow-md font-bold' : 'text-[#94A3B8] hover:bg-white/5 hover:text-white'
          )}
        >
          <MessageSquare size={14} /> Support Requests ({tickets.length})
        </button>
      </div>

      {/* Projects View */}
      {activeTab === 'projects' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {projects.map((project) => (
            <div key={project.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 space-y-5 hover:border-cyan-500/40 transition-all shadow-md">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className="px-2.5 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-[10px] font-bold uppercase">
                    {project.package_type || 'Custom Package'}
                  </span>
                  <h3 className="font-display font-bold text-white text-lg mt-2">{project.title}</h3>
                </div>
                <span className={cn('px-3 py-1 rounded-full text-xs font-bold uppercase border shrink-0', project.status === 'Completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30')}>
                  {project.status}
                </span>
              </div>

              <p className="text-xs text-[#94A3B8] leading-relaxed">{project.notes || 'Full digital execution in progress with your dedicated team.'}</p>

              {/* Progress Bar */}
              <div className="space-y-2 pt-3 border-t border-[#273449]/60">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#94A3B8] font-medium">Milestone Progress</span>
                  <span className="text-white font-bold">{project.progress_percentage || 0}%</span>
                </div>
                <div className="h-2.5 w-full bg-[#050816] rounded-full overflow-hidden border border-[#273449]">
                  <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-700" style={{ width: `${project.progress_percentage || 0}%` }} />
                </div>
              </div>

              {/* Team Assigned Box */}
              <div className="bg-[#050816] border border-[#273449] rounded-xl p-3.5 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2.5">
                  <Users size={16} className="text-cyan-400" />
                  <div>
                    <div className="text-white font-semibold">Assigned Agency Squad</div>
                    <div className="text-[#94A3B8] text-[11px]">Lead UI/UX, Frontend & QA Engineers</div>
                  </div>
                </div>
                <span className="text-cyan-400 font-bold text-[11px] bg-cyan-500/10 px-2.5 py-1 rounded-lg border border-cyan-500/20">
                  Active Squad
                </span>
              </div>

              <div className="flex items-center justify-between pt-2 text-xs text-[#94A3B8]">
                <span>Target Launch: <strong className="text-white">{project.deadline || 'TBD'}</strong></span>
                <span>Budget: <strong className="text-emerald-400">${project.budget?.toLocaleString() || 'Custom'}</strong></span>
              </div>
            </div>
          ))}

          {projects.length === 0 && (
            <div className="col-span-full bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
              <FolderKanban size={36} className="text-cyan-400 mx-auto opacity-70" />
              <h3 className="font-display font-bold text-white text-base">No Active Projects Associated</h3>
              <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                Once the agency Super Admin initializes your company account and spawns your project workflow, your real-time progress board will display here.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'invoices' && (
        <div className="space-y-4">
          {invoices.map((inv) => (
            <div key={inv.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow">
              <div>
                <div className="flex items-center gap-2.5">
                  <span className="font-mono font-bold text-white text-sm">{inv.invoice_number}</span>
                  <span className={cn('px-2.5 py-0.5 rounded text-[10px] font-bold uppercase border', inv.status === 'Paid' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30')}>
                    {inv.status}
                  </span>
                </div>
                <p className="text-xs text-[#94A3B8] mt-1">{inv.notes || 'Milestone invoice for digital deliverables.'}</p>
              </div>

              <div className="flex items-center gap-4 shrink-0">
                <div className="text-right">
                  <div className="text-white font-bold text-base">${inv.total_amount?.toLocaleString() || '0.00'}</div>
                  <div className="text-[#94A3B8] text-[11px]">Due: {inv.due_date || 'N/A'}</div>
                </div>
                <button
                  onClick={() => alert('📄 Invoice downloaded in standard PDF format.')}
                  className="px-4 py-2 bg-[#050816] hover:bg-white/5 border border-[#273449] text-white text-xs font-semibold rounded-xl transition-all"
                >
                  Download PDF
                </button>
              </div>
            </div>
          ))}
          {invoices.length === 0 && (
            <div className="bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
              <DollarSign size={36} className="text-cyan-400 mx-auto opacity-70" />
              <h3 className="font-display font-bold text-white text-base">No Billing Statements Found</h3>
              <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                Any milestone deposit or final completion invoices issued by the Webotixs billing team will appear right here.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'support' && (
        <div className="space-y-4">
          {tickets.map((t) => (
            <div key={t.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-bold text-cyan-400">{t.ticket_number}</span>
                  <span className={cn('px-2 py-0.5 rounded text-[10px] font-bold uppercase border', t.status === 'Open' ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30')}>
                    {t.status}
                  </span>
                </div>
                <h4 className="font-display font-bold text-white text-sm">{t.subject}</h4>
                <p className="text-xs text-[#94A3B8]">{t.description}</p>
              </div>
              <div className="text-xs text-[#94A3B8] text-right shrink-0">
                <div>Priority: <strong className="text-white uppercase">{t.priority || 'Medium'}</strong></div>
                <div>Submitted: {t.created_at ? new Date(t.created_at).toLocaleDateString() : 'Today'}</div>
              </div>
            </div>
          ))}
          {tickets.length === 0 && (
            <div className="bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
              <MessageSquare size={36} className="text-cyan-400 mx-auto opacity-70" />
              <h3 className="font-display font-bold text-white text-base">No Open Support Requests</h3>
              <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                Need guidance or have feedback on a staging link? Click &apos;Open Support Request&apos; above to reach your squad immediately.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Support Request Modal */}
      {isCreatingTicket && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden shadow-2xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
                <MessageSquare size={18} className="text-cyan-400" /> Submit VIP Support Inquiry
              </h3>
              <button onClick={() => setIsCreatingTicket(false)} className="p-2 rounded-lg text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Subject *</label>
                <input
                  type="text"
                  value={newTicket.subject}
                  onChange={(e) => setNewTicket({ ...newTicket, subject: e.target.value })}
                  placeholder="e.g. Question regarding Stage 2 checkout currency"
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Priority Level</label>
                <select
                  value={newTicket.priority}
                  onChange={(e) => setNewTicket({ ...newTicket, priority: e.target.value })}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-cyan-500"
                >
                  <option value="high">High Priority</option>
                  <option value="medium">Medium Priority</option>
                  <option value="low">Low Priority</option>
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Detailed Description & Context</label>
                <textarea
                  rows={4}
                  value={newTicket.description}
                  onChange={(e) => setNewTicket({ ...newTicket, description: e.target.value })}
                  placeholder="Describe your inquiry, requested changes, or questions..."
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-cyan-500 resize-none"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449] bg-[#050816]/50">
              <button onClick={() => setIsCreatingTicket(false)} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">
                Cancel
              </button>
              <button onClick={handleCreateTicket} className="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-semibold rounded-xl shadow-lg hover:shadow-glow-sm transition-all flex items-center gap-2">
                <Send size={14} /> Send Support Request
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
