'use client'

import React, { Suspense } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Sparkles, LogOut, Briefcase, CheckCircle2 } from 'lucide-react'

export default function TeamPortalLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()

  const handleSignOut = () => {
    document.cookie = 'webotixs_admin_session=; path=/; max-age=0'
    document.cookie = 'webotixs_team_session=; path=/; max-age=0'
    router.push('/admin/login')
  }

  return (
    <div className="min-h-screen bg-[#050816] text-white flex flex-col font-sans selection:bg-purple-500/30">
      {/* Standalone Team Staff Top Bar */}
      <header className="sticky top-0 z-50 bg-[#0D1224]/90 backdrop-blur-xl border-b border-[#273449] px-4 md:px-8 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 bg-gradient-to-br from-purple-600 to-blue-500 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20 group-hover:scale-105 transition-all">
                <Briefcase size={18} className="text-white" />
              </div>
              <div>
                <span className="font-display font-bold text-lg tracking-tight text-white flex items-center gap-2">
                  Webotixs
                  <span className="px-2 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 text-[10px] font-bold uppercase tracking-wider">
                    Agency Team Workspace
                  </span>
                </span>
                <p className="text-[11px] text-[#94A3B8] -mt-0.5">Assigned Deliverable Pipeline & Execution Center</p>
              </div>
            </Link>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-[#050816] border border-[#273449] rounded-xl text-xs text-[#94A3B8]">
              <CheckCircle2 size={14} className="text-purple-400" />
              <span>Department Isolation Enabled</span>
            </div>

            <button
              onClick={handleSignOut}
              className="flex items-center gap-2 px-3.5 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-semibold rounded-xl transition-all"
              title="Sign Out of Team Workspace"
            >
              <LogOut size={14} />
              <span className="hidden sm:inline">Sign Out</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area — No Admin Sidebar */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 md:px-8 py-8">
        <Suspense fallback={<div className="text-center py-20 text-[#94A3B8]">Loading department workspace...</div>}>
          {children}
        </Suspense>
      </main>

      {/* Portal Footer */}
      <footer className="border-t border-[#273449] bg-[#0D1224]/40 py-6 px-4 md:px-8 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#94A3B8]">
          <div>© {new Date().getFullYear()} Webotixs Agency — Internal Team Workspace.</div>
          <div className="flex items-center gap-4 text-purple-400 font-medium">
            Automated Task Unlocks & QA Pipeline
          </div>
        </div>
      </footer>
    </div>
  )
}
