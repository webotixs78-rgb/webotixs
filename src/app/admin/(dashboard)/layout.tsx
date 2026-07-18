'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import {
  LayoutDashboard,
  Briefcase,
  FolderKanban,
  FileText,
  Image as ImageIcon,
  Users,
  Building,
  Settings,
  LogOut,
  Menu,
  X,
  Compass,
  MessageSquareCode,
  History,
  Database,
  Sparkles,
  Search,
  Sliders,
  MessageSquareQuote,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface NavSection {
  title: string
  items: {
    label: string
    href: string
    icon: React.ComponentType<{ size?: number; className?: string }>
  }[]
}

const navSections: NavSection[] = [
  {
    title: 'OVERVIEW',
    items: [
      { label: 'Dashboard', href: '/admin/dashboard', icon: LayoutDashboard },
    ],
  },
  {
    title: 'HOMEPAGE CMS',
    items: [
      { label: 'Homepage CMS', href: '/admin/homepage', icon: Sparkles },
    ],
  },
  {
    title: 'CONTENT',
    items: [
      { label: 'Services Manager', href: '/admin/services', icon: Briefcase },
      { label: 'Works / Portfolio', href: '/admin/portfolio', icon: FolderKanban },
      { label: 'Blog Manager', href: '/admin/blogs', icon: FileText },
      { label: 'Media Library', href: '/admin/media', icon: ImageIcon },
      { label: 'Testimonials Manager', href: '/admin/testimonials', icon: MessageSquareQuote },
      { label: 'Industries Manager', href: '/admin/industries', icon: Building },
      { label: 'Team Members', href: '/admin/team', icon: Users },
    ],
  },
  {
    title: 'CRM & JOBS',
    items: [
      { label: 'CRM / Inquiries', href: '/admin/crm', icon: MessageSquareCode },
      { label: 'Careers & Jobs', href: '/admin/careers', icon: Briefcase },
    ],
  },
  {
    title: 'SEO',
    items: [
      { label: 'SEO Settings', href: '/admin/settings?tab=seo', icon: Search },
    ],
  },
  {
    title: 'SYSTEM',
    items: [
      { label: 'Settings', href: '/admin/settings', icon: Sliders },
      { label: 'Activity Logs', href: '/admin/logs', icon: History },
      { label: 'Backup & Export', href: '/admin/backup', icon: Database },
    ],
  },
]

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const pathname = usePathname()
  const supabase = createClient()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [userEmail, setUserEmail] = useState<string>('webotixs78@gmail.com')
  const [unauthorizedRole, setUnauthorizedRole] = useState<string | null>(null)

  useEffect(() => {
    const checkRoleSession = async () => {
      try {
        const storedSession = localStorage.getItem('webotixs_active_session')
        if (storedSession) {
          const parsed = JSON.parse(storedSession)
          if (parsed && parsed.role && parsed.role !== 'Super Admin' && parsed.role !== 'Admin') {
            setUnauthorizedRole(parsed.role)
            return
          }
          if (parsed && parsed.email) setUserEmail(parsed.email)
        }

        const {
          data: { user },
        } = await supabase.auth.getUser()
        if (user && user.email) {
          setUserEmail(user.email)
        }
      } catch {
        setUserEmail('webotixs78@gmail.com')
      }
    }
    checkRoleSession()
  }, [supabase])

  const handleLogout = async () => {
    document.cookie = 'webotixs_admin_session=; path=/; max-age=0'
    document.cookie = 'webotixs_role_session=; path=/; max-age=0'
    document.cookie = 'webotixs_user_id=; path=/; max-age=0'
    localStorage.removeItem('webotixs_active_session')
    try {
      await supabase.auth.signOut()
    } catch {}
    router.push('/admin/login')
    router.refresh()
  }

  if (unauthorizedRole) {
    let returnUrl = '/team/dashboard'
    if (unauthorizedRole === 'Team Manager') returnUrl = '/manager/dashboard'
    else if (unauthorizedRole === 'Client') returnUrl = '/client/dashboard'

    return (
      <div className="min-h-screen bg-[#050816] flex flex-col items-center justify-center p-6 text-center font-sans">
        <div className="bg-[#0D1224] border border-red-500/30 rounded-3xl p-8 max-w-md w-full shadow-2xl space-y-5">
          <div className="w-16 h-16 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center justify-center mx-auto text-red-400">
            <Sliders size={28} />
          </div>
          <div className="space-y-1.5">
            <span className="px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-bold uppercase tracking-wider">
              403 Forbidden — Route Protected
            </span>
            <h1 className="font-display text-xl font-bold text-white mt-2">Access Denied</h1>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              You are currently logged in with the role <strong className="text-white">{unauthorizedRole}</strong>. Super Admin dashboard access is restricted to agency administrators only.
            </p>
          </div>
          <div className="pt-2">
            <Link
              href={returnUrl}
              className="block w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-xl hover:shadow-glow-sm transition-all"
            >
              Return to My {unauthorizedRole} Dashboard
            </Link>
            <button
              onClick={handleLogout}
              className="mt-3 text-xs text-[#94A3B8] hover:text-white underline transition-colors"
            >
              Log Out and Sign In with Different Account
            </button>
          </div>
        </div>
      </div>
    )
  }

  const renderNavList = (onItemClick?: () => void) => (
    <div className="space-y-6">
      {navSections.map((section) => (
        <div key={section.title} className="space-y-1">
          <div className="px-3 text-[10px] font-bold text-[#94A3B8]/60 uppercase tracking-wider mb-2">
            {section.title}
          </div>
          {section.items.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href.split('?')[0]
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onItemClick}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all',
                  isActive
                    ? 'bg-blue-600 text-white shadow-glow-sm'
                    : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
                )}
              >
                <Icon size={16} className={cn(isActive ? 'text-white' : 'text-blue-400')} />
                {item.label}
              </Link>
            )
          })}
        </div>
      ))}
    </div>
  )

  return (
    <div className="min-h-screen bg-[#050816] text-[#F8FAFC] flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex flex-col w-64 border-r border-[#273449] bg-[#0D1224] flex-shrink-0">
        <div className="h-20 flex items-center px-6 border-b border-[#273449]">
          <Link href="/admin/dashboard" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-glow-sm">
              <span className="font-display font-bold text-xs text-white">W</span>
            </div>
            <div>
              <span className="font-display font-bold text-base text-white group-hover:text-blue-400 transition-colors">Webotixs Admin</span>
              <div className="text-[9px] text-emerald-400 font-mono tracking-wider">v0.1.0 PRO</div>
            </div>
          </Link>
        </div>

        {/* Sidebar Nav */}
        <nav className="flex-1 px-4 py-5 overflow-y-auto custom-scrollbar">
          {renderNavList()}
        </nav>

        {/* Footer/User */}
        <div className="p-4 border-t border-[#273449] bg-[#050816]/30">
          <div className="flex items-center gap-3 mb-3 px-2">
            <div className="w-8 h-8 rounded-lg bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400 text-xs font-bold uppercase flex-shrink-0">
              {userEmail[0]}
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-bold truncate text-white">{userEmail}</div>
              <div className="text-[9px] text-[#94A3B8] uppercase tracking-wider font-semibold">Super Admin</div>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center justify-center gap-2 w-full py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-400 text-xs font-semibold rounded-xl transition-colors"
          >
            <LogOut size={13} /> Log Out
          </button>
        </div>
      </aside>

      {/* Mobile Drawer */}
      <div
        className={cn(
          'fixed inset-0 z-50 lg:hidden transition-all duration-300',
          sidebarOpen ? 'visible opacity-100' : 'invisible opacity-0'
        )}
      >
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => setSidebarOpen(false)} />
        <aside
          className={cn(
            'absolute top-0 left-0 bottom-0 w-64 bg-[#0D1224] border-r border-[#273449] flex flex-col transition-transform duration-300',
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          )}
        >
          <div className="h-20 flex items-center justify-between px-6 border-b border-[#273449]">
            <Link href="/admin/dashboard" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
                <span className="font-display font-bold text-xs text-white">W</span>
              </div>
              <span className="font-display font-bold text-base text-white">Webotixs Admin</span>
            </Link>
            <button onClick={() => setSidebarOpen(false)} className="text-[#94A3B8] hover:text-white">
              <X size={20} />
            </button>
          </div>

          <nav className="flex-1 px-4 py-5 overflow-y-auto">
            {renderNavList(() => setSidebarOpen(false))}
          </nav>

          <div className="p-4 border-t border-[#273449]">
            <button
              onClick={handleLogout}
              className="flex items-center justify-center gap-2 w-full py-2.5 bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold rounded-xl"
            >
              <LogOut size={14} /> Log Out
            </button>
          </div>
        </aside>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Topbar */}
        <header className="h-20 border-b border-[#273449] bg-[#0D1224]/80 backdrop-blur-xl flex items-center justify-between px-4 sm:px-8 sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-xl text-[#94A3B8] hover:text-white hover:bg-white/5"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>

            <div className="flex items-center gap-2 text-xs text-[#94A3B8] font-medium">
              <Compass size={14} className="text-blue-500" />
              <span>Admin Panel</span>
              <span>/</span>
              <span className="text-white capitalize font-semibold">{pathname.split('/').pop() || 'Dashboard'}</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="/"
              target="_blank"
              className="px-4 py-2 rounded-xl bg-[#050816] border border-[#273449] text-xs font-semibold text-blue-400 hover:text-white hover:border-blue-500/40 transition-colors flex items-center gap-1.5"
            >
              <span>View Live Website</span>
              <span className="text-[10px]">↗</span>
            </Link>
          </div>
        </header>

        {/* Page Area */}
        <main className="flex-1 p-4 sm:p-8 overflow-x-hidden">{children}</main>
      </div>
    </div>
  )
}
