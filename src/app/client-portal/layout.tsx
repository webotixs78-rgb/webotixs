'use client'

import React, { Suspense } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Sparkles, LogOut, Shield, Bell, CheckCircle2 } from 'lucide-react'
import { useTheme } from '@/components/providers/ThemeProvider'

export default function ClientPortalLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()

  const handleSignOut = () => {
    document.cookie = 'webotixs_admin_session=; path=/; max-age=0'
    document.cookie = 'webotixs_client_session=; path=/; max-age=0'
    router.push('/admin/login')
  }

  return (
    <div className="min-h-screen bg-[#050816] text-white flex flex-col font-sans selection:bg-blue-500/30">
      {/* Standalone VIP Client Top Bar */}
      <header className="sticky top-0 z-50 bg-[#0D1224]/90 backdrop-blur-xl border-b border-[#273449] px-4 md:px-8 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-all">
                <Sparkles size={18} className="text-white" />
              </div>
              <div>
                <span className="font-display font-bold text-lg tracking-tight text-white flex items-center gap-2">
                  Webotixs
                  <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-[10px] font-bold uppercase tracking-wider">
                    VIP Client Portal
                  </span>
                </span>
                <p className="text-[11px] text-[#94A3B8] -mt-0.5">Secure External Dashboard & Deliverable Access</p>
              </div>
            </Link>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-[#050816] border border-[#273449] rounded-xl text-xs text-[#94A3B8]">
              <CheckCircle2 size={14} className="text-emerald-400" />
              <span>Verified Client Connection</span>
            </div>

            <button
              onClick={handleSignOut}
              className="flex items-center gap-2 px-3.5 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-semibold rounded-xl transition-all"
              title="Sign Out of Client Portal"
            >
              <LogOut size={14} />
              <span className="hidden sm:inline">Sign Out</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area — No Admin Sidebar */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 md:px-8 py-8">
        <Suspense fallback={<div className="text-center py-20 text-[#94A3B8]">Loading your project dashboard...</div>}>
          {children}
        </Suspense>
      </main>

      {/* Portal Footer */}
      <footer className="border-t border-[#273449] bg-[#0D1224]/40 py-6 px-4 md:px-8 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#94A3B8]">
          <div>© {new Date().getFullYear()} Webotixs Agency — All rights reserved. Confidential Client Portal.</div>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" /> 256-Bit SSL Encrypted
            </span>
          </div>
        </div>
      </footer>
    </div>
  )
}
