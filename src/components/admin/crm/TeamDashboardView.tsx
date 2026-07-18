'use client'

import React, { useState } from 'react'
import { CheckCircle2, Clock, AlertCircle, Upload, ExternalLink, MessageSquare, FolderKanban, FileText, User, Shield, ArrowRight, Lock } from 'lucide-react'
import { cn, formatDate } from '@/lib/utils'
import { KanbanBoard, CRMTaskItem } from './KanbanBoard'
import { CRMRole } from './CRMRoleSwitcher'

interface TeamDashboardViewProps {
  tasks: CRMTaskItem[]
  projects: any[]
  currentRole: CRMRole
  onOpenTaskDetail: (task: CRMTaskItem) => void
  onTaskStatusChange: (taskId: string, newStatus: CRMTaskItem['status']) => void
  onCompleteTask: (taskId: string, deliverableUrl: string, notes: string) => Promise<void>
}

export function TeamDashboardView({
  tasks,
  projects,
  currentRole,
  onOpenTaskDetail,
  onTaskStatusChange,
  onCompleteTask,
}: TeamDashboardViewProps) {
  const [activeTab, setActiveTab] = useState<'my-tasks' | 'kanban' | 'files' | 'notifications'>('my-tasks')
  const [taskFilter, setTaskFilter] = useState<'today' | 'upcoming' | 'completed'>('today')

  // Filter tasks relevant to current role (or show general team tasks if role matches)
  const roleTasks = tasks.filter((t) => {
    if (currentRole === 'UI/UX Designer') return t.role_required === 'UI/UX Designer'
    if (currentRole === 'WordPress Developer' || currentRole === 'Frontend Developer') return t.role_required.includes('Developer')
    if (currentRole === 'SEO Specialist') return t.role_required === 'SEO Specialist'
    if (currentRole === 'QA Tester') return t.role_required === 'QA Tester'
    return true
  })

  const todayTasks = roleTasks.filter((t) => t.status === 'Todo' || t.status === 'In Progress' || t.status === 'Review')
  const completedTasks = roleTasks.filter((t) => t.status === 'Completed')
  const lockedTasks = roleTasks.filter((t) => t.status === 'Locked')

  const displayedTasks =
    taskFilter === 'today'
      ? todayTasks
      : taskFilter === 'upcoming'
      ? lockedTasks
      : completedTasks

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-blue-900/40 via-purple-900/20 to-[#0D1224] border border-[#273449] rounded-2xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 shadow-glow-sm">
            <User size={24} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display font-bold text-white text-xl">Team Member Workspace</h2>
              <span className="px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-bold uppercase tracking-wider">
                Role: {currentRole}
              </span>
            </div>
            <p className="text-[#94A3B8] text-xs mt-1">
              You are viewing tasks strictly assigned to your department. When you submit deliverables and click Complete, the next workflow step unlocks automatically.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-[#050816] border border-[#273449] text-center min-w-[100px]">
            <div className="text-[10px] font-bold text-[#94A3B8] uppercase">Active Tasks</div>
            <div className="font-display font-bold text-xl text-white mt-0.5">{todayTasks.length}</div>
          </div>
          <div className="p-3 rounded-xl bg-[#050816] border border-[#273449] text-center min-w-[100px]">
            <div className="text-[10px] font-bold text-[#94A3B8] uppercase">Completed</div>
            <div className="font-display font-bold text-xl text-emerald-400 mt-0.5">{completedTasks.length}</div>
          </div>
        </div>
      </div>

      {/* Nav Tabs */}
      <div className="flex items-center justify-between border-b border-[#273449] pb-3 overflow-x-auto gap-2">
        <div className="flex items-center gap-2">
          {[
            { id: 'my-tasks', label: 'Assigned Tasks', icon: CheckCircle2, count: roleTasks.length },
            { id: 'kanban', label: 'Team Kanban Board', icon: FolderKanban, count: tasks.length },
            { id: 'files', label: 'Uploaded Deliverables', icon: FileText, count: roleTasks.filter((t) => t.deliverable_url).length },
          ].map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
                  isActive
                    ? 'bg-blue-600 text-white shadow-glow-sm'
                    : 'bg-[#0D1224] border border-[#273449] text-[#94A3B8] hover:text-white hover:border-[#273449]/80'
                )}
              >
                <Icon size={14} />
                {tab.label}
                <span className={cn('px-1.5 py-0.2 rounded-full text-[10px] font-bold', isActive ? 'bg-black/20 text-white' : 'bg-[#050816] text-[#94A3B8]')}>
                  {tab.count}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* TAB 1: MY TASKS */}
      {activeTab === 'my-tasks' && (
        <div className="space-y-6">
          {/* Sub-filters (`Today's Tasks`, `Upcoming`, `Completed`) */}
          <div className="flex items-center gap-2">
            {[
              { id: 'today', label: "Active / Today's Tasks", count: todayTasks.length, color: 'text-blue-400' },
              { id: 'upcoming', label: 'Upcoming / Locked Steps', count: lockedTasks.length, color: 'text-slate-400' },
              { id: 'completed', label: 'Completed Deliverables', count: completedTasks.length, color: 'text-emerald-400' },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setTaskFilter(f.id as any)}
                className={cn(
                  'px-4 py-2 rounded-xl text-xs font-bold transition-all border flex items-center gap-2',
                  taskFilter === f.id
                    ? 'bg-[#0D1224] border-blue-500 text-white ring-1 ring-blue-500/20'
                    : 'bg-[#050816] border-[#273449]/60 text-[#94A3B8] hover:border-[#273449]'
                )}
              >
                <span>{f.label}</span>
                <span className={cn('px-1.5 py-0.2 rounded bg-[#050816] text-[10px]', f.color)}>{f.count}</span>
              </button>
            ))}
          </div>

          {/* Task Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {displayedTasks.map((task) => (
              <div
                key={task.id}
                onClick={() => onOpenTaskDetail(task)}
                className={cn(
                  'bg-[#0D1224] border rounded-2xl p-5 shadow-xl transition-all cursor-pointer flex flex-col justify-between group',
                  task.status === 'Completed'
                    ? 'border-emerald-500/30 hover:border-emerald-500/60'
                    : task.status === 'Locked'
                    ? 'border-[#273449]/40 opacity-70 bg-[#050816]/60'
                    : 'border-[#273449] hover:border-blue-500/50 hover:shadow-glow-sm'
                )}
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-blue-400 uppercase tracking-wider">Step #{task.step_order} • {task.role_required}</span>
                    <span
                      className={cn(
                        'px-2 py-0.5 rounded text-[10px] font-bold uppercase border',
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

                  <h3 className="font-display font-bold text-white text-base group-hover:text-blue-400 transition-colors">
                    {task.title}
                  </h3>

                  <p className="text-[#94A3B8] text-xs leading-relaxed">{task.description}</p>

                  <div className="pt-2 border-t border-[#273449]/50 flex items-center justify-between text-xs text-[#94A3B8]">
                    <span>📁 Project: <strong className="text-white">{task.project_title || 'Al-Khaleej E-Commerce'}</strong></span>
                    <span className="font-mono text-amber-400">📅 Due: {task.due_date || 'N/A'}</span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-[#273449] flex items-center justify-between">
                  {task.deliverable_url ? (
                    <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-bold">
                      <CheckCircle2 size={15} />
                      Deliverable Link Attached
                    </div>
                  ) : task.status === 'Locked' ? (
                    <div className="flex items-center gap-1.5 text-xs text-slate-400">
                      <Lock size={14} />
                      Waiting for previous step completion
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-xs text-blue-400 font-bold group-hover:translate-x-1 transition-transform">
                      <span>Click to Submit Deliverable ({task.deliverable_type || 'URL'})</span>
                      <ArrowRight size={14} />
                    </div>
                  )}
                </div>
              </div>
            ))}

            {displayedTasks.length === 0 && (
              <div className="col-span-2 py-16 text-center bg-[#0D1224] border border-dashed border-[#273449] rounded-2xl text-[#94A3B8] text-xs">
                No tasks found in this section for {currentRole}.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: KANBAN BOARD */}
      {activeTab === 'kanban' && (
        <KanbanBoard
          tasks={tasks}
          onTaskClick={onOpenTaskDetail}
          onStatusChange={onTaskStatusChange}
        />
      )}

      {/* TAB 3: FILES */}
      {activeTab === 'files' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">My Submitted Deliverables & Files</h3>
            <p className="text-[#94A3B8] text-xs">All Figma prototypes, GitHub repos, and staging links submitted by your department.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {roleTasks
              .filter((t) => t.deliverable_url)
              .map((t) => (
                <div key={t.id} className="p-4 rounded-xl bg-[#050816] border border-[#273449] space-y-2">
                  <div className="flex items-center justify-between text-[10px] font-bold text-blue-400 uppercase">
                    <span>Step #{t.step_order}</span>
                    <span className="text-emerald-400">Verified</span>
                  </div>
                  <h4 className="font-bold text-white text-sm truncate">{t.title}</h4>
                  <a
                    href={t.deliverable_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 text-xs hover:underline flex items-center gap-1.5 truncate"
                  >
                    <ExternalLink size={13} className="shrink-0" />
                    {t.deliverable_url}
                  </a>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
