'use client'

import React, { useState } from 'react'
import { CheckCircle2, Clock, AlertCircle, Lock, User, ExternalLink, ArrowRight, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface CRMTaskItem {
  id: string
  project_id: string
  project_title?: string
  step_order: number
  title: string
  description: string
  role_required: string
  status: 'Todo' | 'In Progress' | 'Review' | 'Completed' | 'Locked'
  due_date?: string
  deliverable_type?: string
  deliverable_url?: string
}

interface KanbanBoardProps {
  tasks: CRMTaskItem[]
  onTaskClick: (task: CRMTaskItem) => void
  onStatusChange?: (taskId: string, newStatus: CRMTaskItem['status']) => void
  readOnly?: boolean
}

const columns: { id: CRMTaskItem['status']; title: string; color: string; border: string; bg: string }[] = [
  { id: 'Todo', title: 'Todo / Assigned', color: 'text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-500/5' },
  { id: 'In Progress', title: 'In Progress', color: 'text-amber-400', border: 'border-amber-500/30', bg: 'bg-amber-500/5' },
  { id: 'Review', title: 'In Review / QA', color: 'text-purple-400', border: 'border-purple-500/30', bg: 'bg-purple-500/5' },
  { id: 'Completed', title: 'Completed', color: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/5' },
  { id: 'Locked', title: 'Locked Step', color: 'text-slate-400', border: 'border-slate-500/30', bg: 'bg-slate-500/5' },
]

export function KanbanBoard({ tasks, onTaskClick, onStatusChange, readOnly }: KanbanBoardProps) {
  const [draggedTaskId, setDraggedTaskId] = useState<string | null>(null)

  const handleDragStart = (e: React.DragEvent, id: string) => {
    setDraggedTaskId(id)
    e.dataTransfer.setData('text/plain', id)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent, status: CRMTaskItem['status']) => {
    e.preventDefault()
    if (!readOnly && onStatusChange && draggedTaskId) {
      if (status !== 'Locked') {
        onStatusChange(draggedTaskId, status)
      }
    }
    setDraggedTaskId(null)
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {columns.map((col) => {
        const colTasks = tasks.filter((t) => t.status === col.id)
        return (
          <div
            key={col.id}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, col.id)}
            className={cn(
              'bg-[#0D1224] border rounded-2xl p-4 flex flex-col min-h-[480px] transition-colors',
              col.border,
              draggedTaskId ? 'border-dashed' : ''
            )}
          >
            <div className="flex items-center justify-between border-b border-[#273449]/60 pb-3 mb-3.5">
              <div className="flex items-center gap-2">
                <div className={cn('w-2 h-2 rounded-full', col.color.replace('text-', 'bg-'))} />
                <h4 className="font-display font-bold text-white text-xs uppercase tracking-wider">{col.title}</h4>
              </div>
              <span className={cn('px-2 py-0.5 rounded-full text-[10px] font-bold', col.bg, col.color)}>
                {colTasks.length}
              </span>
            </div>

            <div className="flex-1 space-y-3 overflow-y-auto pr-1">
              {colTasks.map((task) => (
                <div
                  key={task.id}
                  draggable={!readOnly && task.status !== 'Locked'}
                  onDragStart={(e) => handleDragStart(e, task.id)}
                  onClick={() => onTaskClick(task)}
                  className={cn(
                    'p-3.5 rounded-xl border transition-all cursor-pointer group',
                    task.status === 'Locked'
                      ? 'bg-[#050816]/60 border-[#273449]/40 opacity-70 cursor-not-allowed'
                      : 'bg-[#050816] border-[#273449] hover:border-blue-500/50 hover:shadow-md'
                  )}
                >
                  <div className="flex items-center justify-between text-[10px] text-[#94A3B8] mb-2">
                    <span className="font-bold text-blue-400">Step #{task.step_order}</span>
                    <span className="px-1.5 py-0.5 rounded bg-[#0D1224] border border-[#273449]/80 font-semibold truncate max-w-[110px]">
                      {task.role_required}
                    </span>
                  </div>

                  <h5 className="font-semibold text-white text-xs group-hover:text-blue-400 transition-colors line-clamp-2">
                    {task.title}
                  </h5>

                  <p className="text-[11px] text-[#94A3B8] mt-1 line-clamp-2">{task.description}</p>

                  {task.project_title && (
                    <div className="mt-2.5 pt-2 border-t border-[#273449]/50 text-[10px] text-[#94A3B8] flex items-center justify-between">
                      <span className="truncate max-w-[130px] font-medium text-slate-300">📁 {task.project_title}</span>
                      {task.due_date && <span className="text-amber-400/90 font-mono">📅 {task.due_date}</span>}
                    </div>
                  )}

                  {task.deliverable_url && (
                    <div className="mt-2 flex items-center gap-1.5 text-[10px] text-emerald-400 font-bold bg-emerald-500/10 px-2 py-1 rounded-lg border border-emerald-500/20">
                      <CheckCircle2 size={12} />
                      Deliverable Attached
                    </div>
                  )}

                  {task.status === 'Locked' && (
                    <div className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-400 bg-slate-500/10 px-2 py-1 rounded-lg">
                      <Lock size={12} />
                      Unlocks after previous step
                    </div>
                  )}
                </div>
              ))}

              {colTasks.length === 0 && (
                <div className="flex flex-col items-center justify-center h-48 border border-dashed border-[#273449]/60 rounded-xl text-center p-4">
                  <span className="text-xs text-[#94A3B8]/60">No tasks in this stage</span>
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
