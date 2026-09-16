'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Shield, LogOut, Briefcase, User, CheckCircle2, Sliders } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function TeamDashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [session, setSession] = useState<any | null>(null)
  const [unauthorized, setUnauthorized] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('webotixs_active_session')
    if (stored) {
      const parsed = JSON.parse(stored)
      // Allow Team Member, Team Manager, and Super Admin
      if (parsed.role !== 'Team Member' && parsed.role !== 'Team Manager' && parsed.role !== 'Super Admin' && parsed.role !== 'Admin') {
        setUnauthorized(true)
      } else {
        setSession(parsed)
      }
    } else {
      const cookies = document.cookie
      if (!cookies.includes('webotixs_role_session')) {
        router.push('/admin/login')
      } else {
        setSession({ role: 'Team Member', name: 'Assigned Staff', email: 'staff@webotixs.com', department: 'UI/UX Designer' })
      }
    }
  }, [router])

  const handleLogout = () => {
    localStorage.removeItem('webotixs_active_session')
    document.cookie = 'webotixs_role_session=; path=/; max-age=0'
    document.cookie = 'webotixs_user_id=; path=/; max-age=0'
    document.cookie = 'webotixs_admin_session=; path=/; max-age=0'
    window.location.href = '/admin/login'
  }

  if (unauthorized) {
    return (
      <div className="min-h-screen bg-[#050816] flex flex-col items-center justify-center p-6 text-center font-sans">
        <div className="bg-[#0D1224] border border-red-500/30 rounded-3xl p-8 max-w-md w-full shadow-2xl space-y-5">
          <div className="w-16 h-16 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center justify-center mx-auto text-red-400">
            <Sliders size={28} />
          </div>
          <span className="px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-bold uppercase tracking-wider">
            403 Forbidden — Route Protected
          </span>
          <h1 className="font-display text-xl font-bold text-white mt-2">Staff Portal Access Only</h1>
          <p className="text-xs text-[#94A3B8] leading-relaxed">
            You do not have permission to view this staff dashboard. Please return to your assigned portal.
          </p>
          <button
            onClick={handleLogout}
            className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-xl hover:shadow-glow-sm transition-all"
          >
            Sign Out & Return to Login
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#050816] text-[#E2E8F0] font-sans flex flex-col">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 bg-[#0D1224]/90 backdrop-blur-xl border-b border-[#273449] px-4 md:px-8 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center text-white shadow-md font-display font-bold">
            TM
          </div>
          <div>
            <span className="font-display font-bold text-base text-white tracking-tight">Staff Workspace</span>
            <span className="ml-2.5 px-2 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-[10px] font-bold uppercase">
              {session?.department || 'Employee Board'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 bg-[#050816] px-3.5 py-1.5 rounded-xl border border-[#273449] text-xs">
            <User size={13} className="text-blue-400" />
            <span className="text-[#94A3B8]">Logged in as:</span>
            <span className="text-white font-semibold">{session?.name || session?.email || 'Staff Member'}</span>
          </div>

          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 text-xs font-semibold transition-all"
          >
            <LogOut size={14} /> Sign Out
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 space-y-6">{children}</main>
    </div>
  )
}
