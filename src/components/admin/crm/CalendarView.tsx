'use client'

import React, { useState } from 'react'
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Clock, ExternalLink, Briefcase, CheckCircle2 } from 'lucide-react'
import { cn, formatDate } from '@/lib/utils'
import { CRMTaskItem } from './KanbanBoard'

interface CalendarViewProps {
  tasks: CRMTaskItem[]
  projects?: any[]
  onTaskClick: (task: CRMTaskItem) => void
}

export function CalendarView({ tasks, projects = [], onTaskClick }: CalendarViewProps) {
  const [currentDate, setCurrentDate] = useState(new Date())

  const year = currentDate.getFullYear()
  const month = currentDate.getMonth()

  const firstDayOfMonth = new Date(year, month, 1).getDay()
  const daysInMonth = new Date(year, month + 1, 0).getDate()

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ]

  const prevMonth = () => setCurrentDate(new Date(year, month - 1, 1))
  const nextMonth = () => setCurrentDate(new Date(year, month + 1, 1))

  const getTasksForDay = (day: number) => {
    const dayStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    return tasks.filter((t) => t.due_date && t.due_date.startsWith(dayStr))
  }

  const getProjectsForDay = (day: number) => {
    const dayStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    return projects.filter((p) => p.deadline && p.deadline.startsWith(dayStr))
  }

  const handleGoogleCalendarSync = () => {
    const title = encodeURIComponent('Webotixs Agency CRM Deadlines')
    const details = encodeURIComponent(`Syncing ${tasks.length} tasks and ${projects.length} project deadlines from Webotixs CRM.`)
    window.open(`https://calendar.google.com/calendar/render?action=TEMPLATE&text=${title}&details=${details}`, '_blank')
  }

  return (
    <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <CalendarIcon size={20} />
          </div>
          <div>
            <h3 className="font-display font-bold text-white text-lg">
              {monthNames[month]} {year}
            </h3>
            <p className="text-[#94A3B8] text-xs">Project milestones and workflow task deadlines schedule.</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleGoogleCalendarSync}
            className="flex items-center gap-2 px-4 py-2 bg-[#050816] border border-[#273449] hover:border-blue-500/50 rounded-xl text-xs font-semibold text-white transition-all"
          >
            <ExternalLink size={14} className="text-blue-400" />
            Google Calendar Sync Ready
          </button>

          <div className="flex items-center gap-1 bg-[#050816] border border-[#273449] rounded-xl p-1">
            <button
              onClick={prevMonth}
              className="p-1.5 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={nextMonth}
              className="p-1.5 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-7 gap-px bg-[#273449] rounded-xl overflow-hidden border border-[#273449]">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day) => (
          <div key={day} className="bg-[#0D1224] py-2.5 text-center text-[11px] font-bold text-[#94A3B8] uppercase tracking-wider">
            {day}
          </div>
        ))}

        {Array.from({ length: firstDayOfMonth }).map((_, i) => (
          <div key={`empty-${i}`} className="bg-[#050816]/60 min-h-[110px]" />
        ))}

        {Array.from({ length: daysInMonth }).map((_, i) => {
          const day = i + 1
          const dayTasks = getTasksForDay(day)
          const dayProjects = getProjectsForDay(day)
          const isToday =
            day === new Date().getDate() &&
            month === new Date().getMonth() &&
            year === new Date().getFullYear()

          return (
            <div
              key={`day-${day}`}
              className={cn(
                'bg-[#050816] p-2 min-h-[110px] transition-colors hover:bg-white/[0.02] flex flex-col justify-between',
                isToday ? 'ring-1 ring-blue-500/50 bg-blue-500/[0.02]' : ''
              )}
            >
              <div className="flex items-center justify-between">
                <span
                  className={cn(
                    'w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold',
                    isToday ? 'bg-blue-600 text-white' : 'text-[#94A3B8]'
                  )}
                >
                  {day}
                </span>
                {dayTasks.length + dayProjects.length > 0 && (
                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                    {dayTasks.length + dayProjects.length} due
                  </span>
                )}
              </div>

              <div className="space-y-1 mt-2 flex-1 overflow-y-auto max-h-[85px]">
                {dayProjects.map((proj) => (
                  <div
                    key={proj.id}
                    className="p-1 rounded bg-purple-500/10 border border-purple-500/30 text-[10px] text-purple-300 truncate font-semibold"
                    title={`Project Deadline: ${proj.title}`}
                  >
                    🚀 {proj.title}
                  </div>
                ))}

                {dayTasks.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => onTaskClick(t)}
                    className={cn(
                      'w-full text-left p-1 rounded text-[10px] truncate transition-all font-medium border flex items-center gap-1',
                      t.status === 'Completed'
                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                        : t.status === 'Locked'
                        ? 'bg-slate-500/10 border-slate-500/30 text-slate-400 opacity-60'
                        : 'bg-blue-500/10 border-blue-500/30 text-blue-300 hover:border-blue-500/60'
                    )}
                    title={`Step #${t.step_order}: ${t.title}`}
                  >
                    {t.status === 'Completed' ? <CheckCircle2 size={10} className="shrink-0 text-emerald-400" /> : <Clock size={10} className="shrink-0 text-blue-400" />}
                    <span className="truncate">{t.title}</span>
                  </button>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
