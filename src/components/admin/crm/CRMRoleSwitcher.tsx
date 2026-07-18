'use client'

import React from 'react'
import { Shield, Users, UserCheck, Briefcase, Laptop, Palette, Search, CheckCircle2, Lock, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

export type CRMRole =
  | 'Super Admin'
  | 'Admin'
  | 'Project Manager'
  | 'UI/UX Designer'
  | 'WordPress Developer'
  | 'Frontend Developer'
  | 'Backend Developer'
  | 'SEO Specialist'
  | 'Content Writer'
  | 'QA Tester'
  | 'Client'

interface CRMRoleSwitcherProps {
  currentRole: CRMRole
  onRoleChange: (role: CRMRole) => void
}

const rolesList: { role: CRMRole; label: string; icon: any; color: string; desc: string }[] = [
  { role: 'Super Admin', label: 'Super Admin', icon: Shield, color: 'text-red-400 border-red-500/30 bg-red-500/10', desc: 'Full System, Finance & Settings' },
  { role: 'Project Manager', label: 'Project Manager', icon: Briefcase, color: 'text-purple-400 border-purple-500/30 bg-purple-500/10', desc: 'Team Assignment & Workflows' },
  { role: 'UI/UX Designer', label: 'UI/UX Designer', icon: Palette, color: 'text-pink-400 border-pink-500/30 bg-pink-500/10', desc: 'Design Deliverables & Figma' },
  { role: 'WordPress Developer', label: 'WordPress Dev', icon: Laptop, color: 'text-blue-400 border-blue-500/30 bg-blue-500/10', desc: 'Themes, Plugins & Staging' },
  { role: 'Frontend Developer', label: 'Frontend Dev', icon: Laptop, color: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10', desc: 'React, Next.js & UI Code' },
  { role: 'SEO Specialist', label: 'SEO Specialist', icon: Search, color: 'text-amber-400 border-amber-500/30 bg-amber-500/10', desc: 'Audits, Keywords & Meta' },
  { role: 'QA Tester', label: 'QA Tester', icon: CheckCircle2, color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10', desc: 'Bug Reporting & Sign-off' },
  { role: 'Client', label: 'Client Portal', icon: UserCheck, color: 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10', desc: 'External Client Dashboard View' },
]

export function CRMRoleSwitcher({ currentRole, onRoleChange }: CRMRoleSwitcherProps) {
  return (
    <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-4 shadow-xl">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#273449]/60 pb-3.5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600/10 border border-blue-500/30 flex items-center justify-center text-blue-400 shadow-glow-sm">
            <Sparkles size={20} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-display font-bold text-white text-sm">Role-Based Access Control (RBAC) Simulator</h3>
              <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold uppercase tracking-wider">
                Active
              </span>
            </div>
            <p className="text-[#94A3B8] text-xs mt-0.5">
              Switch roles to verify that Super Admins, Team Members (`Designer`, `Dev`, `SEO`, `QA`), and Clients only access permitted dashboards.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-semibold text-[#94A3B8]">
          <Lock size={14} className="text-blue-400" />
          Current View:
          <span className="text-white font-bold underline decoration-blue-500/50 underline-offset-4">{currentRole}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 pt-3.5">
        {rolesList.map((item) => {
          const Icon = item.icon
          const isActive = currentRole === item.role || (currentRole === 'Admin' && item.role === 'Super Admin')
          return (
            <button
              key={item.role}
              onClick={() => onRoleChange(item.role)}
              className={cn(
                'flex flex-col items-start p-2.5 rounded-xl border transition-all text-left group',
                isActive
                  ? cn(item.color, 'ring-2 ring-blue-500/30 shadow-md')
                  : 'bg-[#050816] border-[#273449]/60 hover:border-[#273449] text-[#94A3B8] hover:text-white'
              )}
            >
              <div className="flex items-center justify-between w-full mb-1.5">
                <Icon size={16} className={cn(isActive ? '' : 'text-[#94A3B8] group-hover:text-blue-400')} />
                {isActive && <div className="w-2 h-2 rounded-full bg-current animate-pulse" />}
              </div>
              <div className="text-xs font-bold leading-tight truncate w-full">{item.label}</div>
              <div className="text-[9px] opacity-70 truncate w-full mt-0.5">{item.desc}</div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
