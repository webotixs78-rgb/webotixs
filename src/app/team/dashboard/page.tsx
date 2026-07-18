'use client'

import React, { useState, useEffect } from 'react'
import { CheckCircle2, Clock, AlertCircle, Upload, MessageSquare, Briefcase, ArrowUpRight, X, Send, User } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function TeamDashboardPage() {
  const [tasks, setTasks] = useState<any[]>([])
  const [projects, setProjects] = useState<any[]>([])
  const [session, setSession] = useState<any | null>(null)
  const [selectedTask, setSelectedTask] = useState<any | null>(null)
  const [deliverableUrl, setDeliverableUrl] = useState('')
  const [deliverableNotes, setDeliverableNotes] = useState('')

  useEffect(() => {
    const storedSession = localStorage.getItem('webotixs_active_session')
    let activeDept = 'UI/UX Designer'
    if (storedSession) {
      const parsed = JSON.parse(storedSession)
      setSession(parsed)
      if (parsed.department) activeDept = parsed.department
    }

    async function loadData() {
      try {
        const pRes = await fetch('/api/crm/projects').catch(() => null)
        if (pRes && pRes.ok) {
          const pData = await pRes.json()
          if (pData.projects && pData.projects.length > 0) {
            setProjects(pData.projects)
          }
        }
      } catch (e) {}

      const localProjects = localStorage.getItem('webotixs_crm_projects')
      const localTasks = localStorage.getItem('webotixs_crm_tasks')
      if (localProjects) setProjects(JSON.parse(localProjects))
      if (localTasks) {
        const allTasks = JSON.parse(localTasks)
        // Filter strictly for this staff member's department or assigned_to
        const staffTasks = allTasks.filter(
          (t: any) =>
            !t.role_required ||
            t.role_required?.toLowerCase() === activeDept?.toLowerCase() ||
            t.assigned_to?.toLowerCase() === activeDept?.toLowerCase() ||
            activeDept === 'Super Admin'
        )
        setTasks(staffTasks)
      }
    }
    loadData()
  }, [])

  const handleStatusChange = (taskId: string, newStatus: string) => {
    const updated = tasks.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t))
    setTasks(updated)

    // Update global localStorage
    const localAll = localStorage.getItem('webotixs_crm_tasks')
    if (localAll) {
      const allTasks = JSON.parse(localAll)
      const synced = allTasks.map((t: any) => (t.id === taskId ? { ...t, status: newStatus } : t))
      localStorage.setItem('webotixs_crm_tasks', JSON.stringify(synced))
    }
  }

  const handleSubmitWork = () => {
    if (!selectedTask) return
    if (!deliverableUrl) {
      alert('Please provide a deliverable link or file URL (e.g., Figma, GitHub, Vercel staging).')
      return
    }

    const taskId = selectedTask.id
    const updated = tasks.map((t) =>
      t.id === taskId
        ? {
            ...t,
            status: 'Review',
            deliverable_url: deliverableUrl,
            deliverable_notes: deliverableNotes,
            submitted_at: new Date().toISOString(),
          }
        : t
    )

    setTasks(updated)
    const localAll = localStorage.getItem('webotixs_crm_tasks')
    if (localAll) {
      const allTasks = JSON.parse(localAll)
      const synced = allTasks.map((t: any) =>
        t.id === taskId
          ? {
              ...t,
              status: 'Review',
              deliverable_url: deliverableUrl,
              deliverable_notes: deliverableNotes,
              submitted_at: new Date().toISOString(),
            }
          : t
      )
      localStorage.setItem('webotixs_crm_tasks', JSON.stringify(synced))
    }

    setSelectedTask(null)
    setDeliverableUrl('')
    setDeliverableNotes('')
    alert('✅ Deliverable Submitted for Manager/Client Review! Task status changed to Review.')
  }

  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="bg-gradient-to-r from-blue-900/40 via-[#0D1224] to-cyan-900/30 border border-blue-500/30 rounded-3xl p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xl">
        <div>
          <span className="px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-bold uppercase tracking-wider">
            Staff Pipeline Board
          </span>
          <h1 className="font-display text-2xl md:text-3xl font-bold text-white tracking-tight mt-2">
            {session?.department || 'Employee'} Task Pipeline
          </h1>
          <p className="text-xs md:text-sm text-[#94A3B8] max-w-xl mt-1">
            Focus strictly on your assigned action items, submit deliverables when finished, and notify your Team Manager.
          </p>
        </div>

        <div className="bg-[#050816]/80 px-4 py-3 rounded-2xl border border-[#273449] text-xs space-y-0.5 shrink-0">
          <div className="text-white font-semibold">Assigned Staff Account</div>
          <div className="text-blue-400 font-mono">{session?.email || 'staff@webotixs.com'}</div>
        </div>
      </div>

      {/* Task Grid */}
      <div className="space-y-4">
        <h2 className="font-display text-lg font-bold text-white flex items-center gap-2">
          <Briefcase size={18} className="text-blue-400" /> My Assigned Deliverables ({tasks.length})
        </h2>

        {tasks.map((task) => (
          <div
            key={task.id}
            className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 hover:border-blue-500/40 transition-all shadow-md"
          >
            <div className="space-y-2 max-w-2xl">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase">
                  Project: {task.project_title || 'Client Project'}
                </span>
                <span
                  className={cn(
                    'px-2.5 py-0.5 rounded text-[10px] font-bold uppercase border',
                    task.status === 'Completed'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                      : task.status === 'Review'
                      ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30'
                      : task.status === 'Locked'
                      ? 'bg-red-500/10 text-red-400 border-red-500/30'
                      : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  )}
                >
                  Status: {task.status}
                </span>
                {task.priority && (
                  <span className="px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-semibold uppercase">
                    Priority: {task.priority}
                  </span>
                )}
              </div>

              <h3 className="font-display font-bold text-white text-base">{task.title}</h3>
              <p className="text-xs text-[#94A3B8] leading-relaxed">{task.description || 'No detailed instructions provided.'}</p>

              {task.deliverable_url && (
                <div className="bg-[#050816] border border-[#273449] rounded-xl p-3 text-xs space-y-1">
                  <div className="text-emerald-400 font-semibold flex items-center gap-1.5">
                    <CheckCircle2 size={14} /> Submitted Deliverable Link:
                  </div>
                  <a href={task.deliverable_url} target="_blank" rel="noreferrer" className="text-cyan-400 underline font-mono break-all block">
                    {task.deliverable_url}
                  </a>
                  {task.deliverable_notes && <p className="text-[#94A3B8] text-[11px] mt-1">Notes: {task.deliverable_notes}</p>}
                </div>
              )}
            </div>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 shrink-0 pt-4 md:pt-0 border-t md:border-t-0 border-[#273449]">
              <div className="text-left md:text-right text-xs mr-2">
                <div className="text-[#94A3B8]">Deadline:</div>
                <div className="text-white font-semibold font-mono">{task.due_date || 'TBD'}</div>
              </div>

              {task.status !== 'Locked' && task.status !== 'Completed' && (
                <div className="flex items-center gap-2">
                  <select
                    value={task.status}
                    onChange={(e) => handleStatusChange(task.id, e.target.value)}
                    className="px-3 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs font-semibold focus:outline-none focus:border-blue-500"
                  >
                    <option value="Todo">Todo</option>
                    <option value="In Progress">In Progress</option>
                    <option value="Review">In Review</option>
                  </select>

                  <button
                    onClick={() => {
                      setSelectedTask(task)
                      setDeliverableUrl(task.deliverable_url || '')
                      setDeliverableNotes(task.deliverable_notes || '')
                    }}
                    className="px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-xl shadow hover:shadow-glow-sm transition-all flex items-center gap-1.5"
                  >
                    <Upload size={14} /> Submit Work
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {tasks.length === 0 && (
          <div className="bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
            <CheckCircle2 size={36} className="text-blue-400 mx-auto opacity-70" />
            <h3 className="font-display font-bold text-white text-base">No Assigned Tasks Available</h3>
            <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
              You currently have no tasks pending in your department pipeline. Once your Team Manager or Super Admin assigns deliverables, they will appear here instantly.
            </p>
          </div>
        )}
      </div>

      {/* Upload Deliverable Modal */}
      {selectedTask && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden shadow-2xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
                <Upload size={18} className="text-blue-400" /> Submit Deliverable for Review
              </h3>
              <button onClick={() => setSelectedTask(null)} className="p-2 rounded-lg text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="bg-[#050816] p-3.5 rounded-2xl border border-[#273449] text-xs">
                <span className="text-[#94A3B8]">Task: </span>
                <strong className="text-white">{selectedTask.title}</strong>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Deliverable URL (Staging Link, Figma, GitHub, Google Doc) *</label>
                <input
                  type="text"
                  value={deliverableUrl}
                  onChange={(e) => setDeliverableUrl(e.target.value)}
                  placeholder="https://..."
                  className="w-full px-4 py-3 bg-[#050816] border border-[#273449] rounded-2xl text-white text-sm focus:outline-none focus:border-blue-500 font-mono"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Completion Notes / Instructions for Manager</label>
                <textarea
                  rows={3}
                  value={deliverableNotes}
                  onChange={(e) => setDeliverableNotes(e.target.value)}
                  placeholder="Explain what was accomplished or specific review notes..."
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-2xl text-white text-sm focus:outline-none focus:border-blue-500 resize-none"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449] bg-[#050816]/50">
              <button onClick={() => setSelectedTask(null)} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">
                Cancel
              </button>
              <button onClick={handleSubmitWork} className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl shadow-lg hover:shadow-glow-sm transition-all flex items-center gap-2">
                <Send size={14} /> Submit & Request Approval
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
