'use client'

import { useState } from 'react'
import {
  History,
  Search,
  Filter,
  Trash2,
  Calendar,
  User,
  Activity,
  CheckCircle2,
  AlertCircle,
  FileText,
  Briefcase,
  Users,
  Image as ImageIcon,
  Shield,
} from 'lucide-react'
import { cn, formatDateShort } from '@/lib/utils'

interface ActivityLog {
  id: string
  action: 'CREATE' | 'UPDATE' | 'DELETE' | 'LOGIN' | 'SETTINGS'
  entity_type: 'service' | 'portfolio' | 'blog' | 'team' | 'media' | 'crm' | 'auth' | 'settings'
  entity_id: string
  details: string
  admin_email: string
  created_at: string
}

const mockLogs: ActivityLog[] = [
  {
    id: '1',
    action: 'CREATE',
    entity_type: 'blog',
    entity_id: 'future-web-design-trends-2025',
    details: 'Published new blog post: "Future Web Design Trends 2025"',
    admin_email: 'ahmed@webotixs.com',
    created_at: new Date(Date.now() - 1000 * 60 * 25).toISOString(), // 25 mins ago
  },
  {
    id: '2',
    action: 'UPDATE',
    entity_type: 'crm',
    entity_id: 'lead-1',
    details: 'Updated CRM lead status to "qualified" for Al-Khaleej Retail Group',
    admin_email: 'sarah@webotixs.com',
    created_at: new Date(Date.now() - 1000 * 60 * 120).toISOString(), // 2 hours ago
  },
  {
    id: '3',
    action: 'CREATE',
    entity_type: 'media',
    entity_id: 'hero-mesh-background.png',
    details: 'Uploaded media asset to bucket folder /Backgrounds',
    admin_email: 'marcus@webotixs.com',
    created_at: new Date(Date.now() - 1000 * 60 * 240).toISOString(), // 4 hours ago
  },
  {
    id: '4',
    action: 'UPDATE',
    entity_type: 'service',
    entity_id: 'web-design-development',
    details: 'Updated pricing tiers and feature tags for Web Design & Development',
    admin_email: 'ahmed@webotixs.com',
    created_at: new Date(Date.now() - 1000 * 60 * 600).toISOString(), // 10 hours ago
  },
  {
    id: '5',
    action: 'LOGIN',
    entity_type: 'auth',
    entity_id: 'ahmed@webotixs.com',
    details: 'Admin user successfully signed in from IP 185.192.x.x (Dubai, UAE)',
    admin_email: 'ahmed@webotixs.com',
    created_at: new Date(Date.now() - 1000 * 60 * 1440).toISOString(), // 1 day ago
  },
  {
    id: '6',
    action: 'DELETE',
    entity_type: 'portfolio',
    entity_id: 'legacy-project-v1',
    details: 'Deleted archived portfolio case study: "Legacy Project v1"',
    admin_email: 'ahmed@webotixs.com',
    created_at: new Date(Date.now() - 1000 * 60 * 2880).toISOString(), // 2 days ago
  },
]

const actionColors: Record<ActivityLog['action'], string> = {
  CREATE: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25',
  UPDATE: 'bg-blue-500/10 text-blue-400 border-blue-500/25',
  DELETE: 'bg-red-500/10 text-red-400 border-red-500/25',
  LOGIN: 'bg-violet-500/10 text-violet-400 border-violet-500/25',
  SETTINGS: 'bg-amber-500/10 text-amber-400 border-amber-500/25',
}

const entityIcons: Record<ActivityLog['entity_type'], React.ComponentType<{ size?: number; className?: string }>> = {
  service: Briefcase,
  portfolio: Briefcase,
  blog: FileText,
  team: Users,
  media: ImageIcon,
  crm: Activity,
  auth: Shield,
  settings: History,
}

export default function AdminLogsPage() {
  const [logs, setLogs] = useState<ActivityLog[]>([...mockLogs])
  const [search, setSearch] = useState('')
  const [actionFilter, setActionFilter] = useState<string>('ALL')
  const [clearing, setClearing] = useState(false)

  const filteredLogs = logs.filter((log) => {
    const matchesAction = actionFilter === 'ALL' || log.action === actionFilter
    const matchesSearch =
      log.details.toLowerCase().includes(search.toLowerCase()) ||
      log.admin_email.toLowerCase().includes(search.toLowerCase()) ||
      log.entity_type.toLowerCase().includes(search.toLowerCase())
    return matchesAction && matchesSearch
  })

  const handleClearLogs = async () => {
    if (!window.confirm('Are you sure you want to clear all archived activity logs?')) return
    setClearing(true)
    await new Promise((r) => setTimeout(r, 600))
    setLogs([])
    setClearing(false)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2.5">
            <History size={24} className="text-blue-500" /> Activity Audit Logs
          </h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Real-time audit trail of admin logins, content edits, media uploads, and CRM updates.
          </p>
        </div>

        {logs.length > 0 && (
          <button
            onClick={handleClearLogs}
            disabled={clearing}
            className="flex items-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold rounded-xl hover:bg-red-500/20 transition-colors disabled:opacity-50"
          >
            <Trash2 size={14} /> Clear Log History
          </button>
        )}
      </div>

      {/* Filters Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-[#0D1224] border border-[#273449] rounded-2xl p-4">
        {/* Search */}
        <div className="relative w-full sm:w-80">
          <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search logs by email, action, details..."
            className="w-full pl-10 pr-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50"
          />
        </div>

        {/* Action Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 w-full sm:w-auto">
          {(['ALL', 'CREATE', 'UPDATE', 'DELETE', 'LOGIN'] as const).map((action) => (
            <button
              key={action}
              onClick={() => setActionFilter(action)}
              className={cn(
                'px-3 py-1.5 rounded-xl text-xs font-bold uppercase tracking-wider border transition-all',
                actionFilter === action
                  ? 'bg-blue-600 text-white border-blue-600 shadow-glow-sm'
                  : 'bg-[#050816] text-[#94A3B8] border-[#273449] hover:border-blue-500/30'
              )}
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      {/* Logs Table */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#273449] text-[#94A3B8] text-xs uppercase tracking-wider bg-[#0A0E1F]">
                <th className="text-left px-6 py-4 font-semibold">Action & Entity</th>
                <th className="text-left px-4 py-4 font-semibold">Details</th>
                <th className="text-left px-4 py-4 font-semibold hidden md:table-cell">Admin User</th>
                <th className="text-right px-6 py-4 font-semibold">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {filteredLogs.map((log) => {
                const Icon = entityIcons[log.entity_type] ?? Activity
                return (
                  <tr key={log.id} className="border-b border-[#273449]/50 hover:bg-white/[0.02] transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-[#050816] border border-[#273449] flex items-center justify-center text-blue-400 flex-shrink-0">
                          <Icon size={16} />
                        </div>
                        <div>
                          <span
                            className={cn(
                              'px-2 py-0.5 rounded-full text-[9px] font-bold uppercase border',
                              actionColors[log.action]
                            )}
                          >
                            {log.action}
                          </span>
                          <div className="text-[10px] text-[#94A3B8] uppercase tracking-wider font-semibold mt-1">
                            {log.entity_type}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-xs font-medium text-white max-w-md leading-relaxed">{log.details}</div>
                      <div className="text-[10px] text-[#94A3B8] font-mono mt-0.5">ID: {log.entity_id}</div>
                    </td>
                    <td className="px-4 py-4 hidden md:table-cell">
                      <div className="flex items-center gap-2 text-xs text-[#94A3B8]">
                        <User size={13} className="text-blue-500" />
                        <span className="text-white font-medium">{log.admin_email}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right text-xs text-[#94A3B8] font-mono">
                      <div className="flex items-center justify-end gap-1.5">
                        <Calendar size={12} className="text-[#94A3B8]/60" />
                        {formatDateShort(log.created_at)}
                      </div>
                      <div className="text-[10px] text-[#94A3B8]/60 mt-0.5">
                        {new Date(log.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {filteredLogs.length === 0 && (
          <div className="py-16 text-center text-[#94A3B8] text-xs">
            No activity logs match your search or filter options.
          </div>
        )}
      </div>
    </div>
  )
}
