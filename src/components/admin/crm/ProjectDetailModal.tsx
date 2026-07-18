'use client'

import React, { useState } from 'react'
import { X, CheckCircle2, Clock, Lock, Upload, Send, MessageSquare, History, FileText, ExternalLink, User, Shield, Briefcase, Paperclip, AlertCircle } from 'lucide-react'
import { cn, formatDate } from '@/lib/utils'
import { CRMTaskItem } from './KanbanBoard'
import { CRMRole } from './CRMRoleSwitcher'

interface ProjectDetailModalProps {
  project: any
  tasks: CRMTaskItem[]
  currentRole: CRMRole
  onClose: () => void
  onCompleteTask: (taskId: string, deliverableUrl: string, notes: string) => Promise<void>
  onSendMessage?: (content: string, attachmentUrl?: string) => Promise<void>
}

export function ProjectDetailModal({
  project,
  tasks,
  currentRole,
  onClose,
  onCompleteTask,
  onSendMessage,
}: ProjectDetailModalProps) {
  const [activeTab, setActiveTab] = useState<'workflow' | 'chat' | 'files' | 'timeline'>('workflow')
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [deliverableInput, setDeliverableInput] = useState('')
  const [notesInput, setNotesInput] = useState('')
  const [chatInput, setChatInput] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const [messages, setMessages] = useState<any[]>([
    {
      id: '1',
      sender_name: 'Ahmed Al-Rashid',
      sender_role: 'Super Admin',
      content: 'Welcome everyone! We have initialized this project with the default agency workflow template. Please review requirements and kick off Step 1.',
      created_at: new Date(Date.now() - 3600000 * 24).toISOString(),
    },
    {
      id: '2',
      sender_name: 'Sarah Chen',
      sender_role: 'UI/UX Designer',
      content: '@Ahmed @Client I have completed the wireframes and moodboard. Figma link is attached in Step 1 deliverables.',
      created_at: new Date(Date.now() - 3600000 * 12).toISOString(),
    },
  ])

  const projectTasks = tasks.filter((t) => t.project_id === project.id)
  const activeTask = projectTasks.find((t) => t.id === selectedTaskId) || projectTasks[0]

  const handleTaskSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!activeTask) return
    setSubmitting(true)
    try {
      await onCompleteTask(activeTask.id, deliverableInput, notesInput)
      setDeliverableInput('')
      setNotesInput('')
    } finally {
      setSubmitting(false)
    }
  }

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!chatInput.trim()) return
    const newMsg = {
      id: Date.now().toString(),
      sender_name: currentRole === 'Client' ? 'Client Contact' : 'Active Team Member',
      sender_role: currentRole,
      content: chatInput,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, newMsg])
    if (onSendMessage) await onSendMessage(chatInput)
    setChatInput('')
  }

  const progress = project.progress_percentage ?? 0

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto animate-in fade-in duration-200">
      <div className="w-full max-w-4xl max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl flex flex-col overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[#273449] flex items-center justify-between bg-[#050816]/50">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-[10px] font-bold uppercase tracking-wider">
                {project.package_type || 'Custom Workflow'}
              </span>
              <span className="px-2 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-[10px] font-bold uppercase tracking-wider">
                Priority: {project.priority || 'medium'}
              </span>
            </div>
            <h2 className="font-display font-bold text-white text-xl mt-1.5">{project.title}</h2>
            <p className="text-[#94A3B8] text-xs mt-0.5">
              Client: <span className="text-white font-semibold">{project.client?.company_name || 'Al-Khaleej Retail'}</span> • Deadline: <span className="text-amber-400 font-mono">{project.deadline}</span>
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-[#050816] border border-[#273449] hover:border-blue-500/50 text-[#94A3B8] hover:text-white transition-all"
          >
            <X size={18} />
          </button>
        </div>

        {/* Progress Bar & Navigation Tabs */}
        <div className="px-6 pt-4 pb-0 border-b border-[#273449] bg-[#0D1224]">
          <div className="mb-4">
            <div className="flex items-center justify-between text-xs mb-1.5 font-semibold">
              <span className="text-[#94A3B8]">Overall Project Completion Rate</span>
              <span className="text-blue-400 font-bold">{progress}% Delivered</span>
            </div>
            <div className="w-full h-2.5 rounded-full bg-[#050816] border border-[#273449] overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 via-cyan-400 to-emerald-400 transition-all duration-700 rounded-full shadow-glow-sm"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          <div className="flex items-center gap-2 overflow-x-auto">
            {[
              { id: 'workflow', label: 'Workflow & Tasks', icon: Clock, count: projectTasks.length },
              { id: 'chat', label: 'Internal Chat & @Mentions', icon: MessageSquare, count: messages.length },
              { id: 'files', label: 'Deliverables & Files', icon: FileText, count: projectTasks.filter((t) => t.deliverable_url).length },
              { id: 'timeline', label: 'Audit Timeline', icon: History },
            ].map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={cn(
                    'flex items-center gap-2 px-4 py-3 border-b-2 font-semibold text-xs transition-all whitespace-nowrap',
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-400 bg-blue-500/5'
                      : 'border-transparent text-[#94A3B8] hover:text-white'
                  )}
                >
                  <Icon size={14} />
                  {tab.label}
                  {tab.count !== undefined && (
                    <span className="px-1.5 py-0.2 rounded-full bg-[#050816] text-[10px] font-bold text-white border border-[#273449]">
                      {tab.count}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* TAB 1: WORKFLOW */}
          {activeTab === 'workflow' && (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
              {/* Steps sidebar list */}
              <div className="md:col-span-5 space-y-2.5 pr-2">
                <h4 className="text-xs font-bold text-[#94A3B8] uppercase tracking-wider mb-3">Lifecycle Stages ({projectTasks.length})</h4>
                {projectTasks.map((task) => {
                  const isSelected = activeTask?.id === task.id
                  return (
                    <div
                      key={task.id}
                      onClick={() => setSelectedTaskId(task.id)}
                      className={cn(
                        'p-3.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between',
                        isSelected
                          ? 'bg-blue-600/10 border-blue-500 text-white shadow-md'
                          : task.status === 'Locked'
                          ? 'bg-[#050816]/40 border-[#273449]/40 opacity-60 text-[#94A3B8]'
                          : 'bg-[#050816] border-[#273449] hover:border-[#273449]/80 text-[#94A3B8] hover:text-white'
                      )}
                    >
                      <div className="flex items-center gap-3">
                        {task.status === 'Completed' && <CheckCircle2 size={18} className="text-emerald-400 shrink-0" />}
                        {task.status === 'Locked' && <Lock size={16} className="text-slate-400 shrink-0" />}
                        {task.status !== 'Completed' && task.status !== 'Locked' && <Clock size={16} className="text-blue-400 shrink-0" />}
                        <div>
                          <div className="text-[10px] font-bold uppercase tracking-wider text-blue-400">Step #{task.step_order}</div>
                          <div className="text-xs font-semibold text-white leading-tight mt-0.5">{task.title}</div>
                        </div>
                      </div>

                      <span
                        className={cn(
                          'px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider border',
                          task.status === 'Completed'
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                            : task.status === 'Locked'
                            ? 'bg-slate-500/10 text-slate-400 border-slate-500/30'
                            : 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                        )}
                      >
                        {task.status}
                      </span>
                    </div>
                  )
                })}
              </div>

              {/* Active Step details & completion form */}
              <div className="md:col-span-7 bg-[#050816] border border-[#273449] rounded-2xl p-5 space-y-5">
                {activeTask ? (
                  <>
                    <div className="border-b border-[#273449] pb-4">
                      <div className="flex items-center justify-between text-xs font-bold text-[#94A3B8] mb-1">
                        <span className="text-blue-400 uppercase tracking-wider">Step #{activeTask.step_order} • {activeTask.role_required}</span>
                        <span>Due: {activeTask.due_date || 'N/A'}</span>
                      </div>
                      <h3 className="font-display font-bold text-white text-lg">{activeTask.title}</h3>
                      <p className="text-[#94A3B8] text-xs leading-relaxed mt-2">{activeTask.description}</p>
                    </div>

                    {/* Deliverable info */}
                    {activeTask.deliverable_url && (
                      <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-between">
                        <div>
                          <div className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">✅ Submitted Deliverable</div>
                          <a
                            href={activeTask.deliverable_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-white font-semibold text-xs hover:underline flex items-center gap-1.5 mt-1 truncate max-w-sm"
                          >
                            {activeTask.deliverable_url}
                            <ExternalLink size={13} />
                          </a>
                        </div>
                        <span className="text-emerald-400 text-xs font-bold">Approved</span>
                      </div>
                    )}

                    {/* Completion Form */}
                    {activeTask.status !== 'Completed' && activeTask.status !== 'Locked' && currentRole !== 'Client' ? (
                      <form onSubmit={handleTaskSubmit} className="space-y-4 pt-2">
                        <div className="space-y-1.5">
                          <label className="text-xs font-bold text-white flex items-center justify-between">
                            <span>Submit Deliverable URL ({activeTask.deliverable_type || 'Figma / Staging Link'})</span>
                            <span className="text-red-400 text-[10px]">* Required to unlock next step</span>
                          </label>
                          <input
                            type="url"
                            required
                            placeholder="https://figma.com/file/..."
                            value={deliverableInput}
                            onChange={(e) => setDeliverableInput(e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50"
                          />
                        </div>

                        <div className="space-y-1.5">
                          <label className="text-xs font-bold text-white">Deliverable Notes / Handoff Summary</label>
                          <textarea
                            rows={3}
                            placeholder="Enter any notes or QA instructions for the next role in the workflow..."
                            value={notesInput}
                            onChange={(e) => setNotesInput(e.target.value)}
                            className="w-full px-4 py-2 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50"
                          />
                        </div>

                        <button
                          type="submit"
                          disabled={submitting}
                          className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs shadow-glow-sm transition-all flex items-center justify-center gap-2"
                        >
                          <CheckCircle2 size={16} />
                          {submitting ? 'Submitting & Unlocking Next Step...' : `Mark Step #${activeTask.step_order} Complete & Auto-Unlock Next Task`}
                        </button>
                      </form>
                    ) : activeTask.status === 'Locked' ? (
                      <div className="p-4 rounded-xl bg-slate-500/10 border border-slate-500/30 text-center space-y-2">
                        <Lock size={20} className="mx-auto text-slate-400" />
                        <h5 className="font-bold text-white text-xs">Step #{activeTask.step_order} is currently Locked</h5>
                        <p className="text-[11px] text-[#94A3B8]">
                          This step will automatically unlock the moment Step #{activeTask.step_order - 1} (`{projectTasks.find((t) => t.step_order === activeTask.step_order - 1)?.title}`) is marked completed.
                        </p>
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-center py-12 text-[#94A3B8] text-xs">Select a workflow task from the left to view requirements.</div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: CHAT */}
          {activeTab === 'chat' && (
            <div className="flex flex-col h-[480px]">
              <div className="flex-1 overflow-y-auto space-y-4 pr-2 mb-4">
                {messages.map((msg) => (
                  <div key={msg.id} className="p-4 rounded-2xl bg-[#050816] border border-[#273449] space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white text-xs">{msg.sender_name}</span>
                        <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-semibold">
                          {msg.sender_role}
                        </span>
                      </div>
                      <span className="text-[10px] text-[#94A3B8]">{formatDate(msg.created_at)}</span>
                    </div>
                    <p className="text-white text-xs leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  </div>
                ))}
              </div>

              <form onSubmit={handleChatSubmit} className="flex gap-2">
                <input
                  type="text"
                  placeholder="Type a message or use @mention (`@Ahmed`, `@Sarah`, `@Client`)..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  className="flex-1 px-4 py-3 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/40 focus:outline-none focus:border-blue-500/50"
                />
                <button
                  type="submit"
                  className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs transition-colors flex items-center gap-2"
                >
                  <Send size={14} />
                  Send
                </button>
              </form>
            </div>
          )}

          {/* TAB 3: FILES */}
          {activeTab === 'files' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {projectTasks
                  .filter((t) => t.deliverable_url)
                  .map((task) => (
                    <div key={task.id} className="p-4 rounded-2xl bg-[#050816] border border-[#273449] space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">Step #{task.step_order} Deliverable</span>
                        <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[9px] font-bold">Verified</span>
                      </div>
                      <h5 className="font-bold text-white text-xs truncate">{task.title} ({task.deliverable_type || 'Link'})</h5>
                      <a
                        href={task.deliverable_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 text-xs hover:underline flex items-center gap-1.5 truncate"
                      >
                        <ExternalLink size={13} className="shrink-0" />
                        {task.deliverable_url}
                      </a>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* TAB 4: TIMELINE */}
          {activeTab === 'timeline' && (
            <div className="space-y-4 border-l-2 border-blue-500/30 pl-4 ml-2">
              {[
                { time: '2026-07-16 10:30', actor: 'Ahmed Al-Rashid', role: 'Super Admin', action: 'Project Created', details: 'Initialized workflow template and auto-generated 5 sequential steps.' },
                { time: '2026-07-16 11:00', actor: 'System Auto-Trigger', role: 'Automation', action: 'Client Credentials Generated', details: 'Sent secure portal credentials via email and unlocked Step 1 (UI/UX Design).' },
                { time: '2026-07-17 14:15', actor: 'Sarah Chen', role: 'UI/UX Designer', action: 'Task Completed', details: 'Submitted Figma prototype link. Automatically unlocked Step 2 (Shopify Custom Development).' },
                { time: '2026-07-18 09:00', actor: 'Zayn Malik', role: 'Frontend Developer', action: 'Task Completed', details: 'Submitted Vercel staging URL. Automatically unlocked Step 3 (Product Import & SEO).' },
              ].map((log, i) => (
                <div key={i} className="relative space-y-1">
                  <div className="absolute -left-[23px] top-1 w-3.5 h-3.5 rounded-full bg-blue-500 border-2 border-[#0D1224]" />
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-bold text-white">{log.action}</span>
                    <span className="text-[#94A3B8] text-[10px]">• {log.actor} ({log.role})</span>
                    <span className="text-slate-400 font-mono text-[10px] ml-auto">{log.time}</span>
                  </div>
                  <p className="text-xs text-[#94A3B8] leading-relaxed">{log.details}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
