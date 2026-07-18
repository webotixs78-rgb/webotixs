'use client'

import React, { useState, useEffect } from 'react'
import { Plus, Search, Filter, Briefcase, FolderKanban, Users, Shield, FileText, Receipt, LifeBuoy, BarChart3, MessageSquareCode, CheckCircle2, Clock, Eye, AlertCircle, Sparkles, ExternalLink, Download, X, Key, Mail, Lock, UserCheck, UserX, Copy, Check, RefreshCw, Sliders } from 'lucide-react'
import { cn, formatDate } from '@/lib/utils'
import { KanbanBoard, CRMTaskItem } from './KanbanBoard'
import { CalendarView } from './CalendarView'
import { CRMRole } from './CRMRoleSwitcher'

interface AdminDashboardViewProps {
  projects: any[]
  tasks: CRMTaskItem[]
  invoices: any[]
  tickets: any[]
  inquiries: any[]
  currentRole: CRMRole
  onOpenProject: (project: any) => void
  onTaskStatusChange: (taskId: string, newStatus: CRMTaskItem['status']) => void
  onCreateProject: (projectData: any) => Promise<void>
}

export function AdminDashboardView({
  projects,
  tasks,
  invoices,
  tickets,
  inquiries,
  currentRole,
  onOpenProject,
  onTaskStatusChange,
  onCreateProject,
}: AdminDashboardViewProps) {
  const [activeTab, setActiveTab] = useState<
    'users' | 'projects' | 'kanban' | 'calendar' | 'team' | 'templates' | 'invoices' | 'tickets' | 'reports' | 'inquiries'
  >('users')

  const [search, setSearch] = useState('')
  const [userSearch, setUserSearch] = useState('')
  const [userRoleFilter, setUserRoleFilter] = useState<'ALL' | 'Team Manager' | 'Team Member' | 'Client'>('ALL')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  // Dynamic User & Credential Generator State
  const [crmUsers, setCrmUsers] = useState<any[]>([])
  const [isUserModalOpen, setIsUserModalOpen] = useState(false)
  const [userModalType, setUserModalType] = useState<'Team Manager' | 'Team Member' | 'Client'>('Team Manager')
  const [editingUser, setEditingUser] = useState<any | null>(null)
  const [userForm, setUserForm] = useState({
    userId: '',
    name: '',
    email: '',
    password: '',
    department: 'UI/UX Design',
    companyName: '',
    phone: '',
    address: '',
    website: '',
    notes: '',
    status: 'active' as 'active' | 'suspended',
  })

  // Load all dynamic users
  useEffect(() => {
    function loadUsers() {
      const storedUsersRaw = localStorage.getItem('webotixs_crm_users')
      const storedTeamRaw = localStorage.getItem('webotixs_team_members')
      const storedClientsRaw = localStorage.getItem('webotixs_client_accounts')

      let combined: any[] = []
      if (storedUsersRaw) {
        try {
          const parsed = JSON.parse(storedUsersRaw)
          if (Array.isArray(parsed)) combined = [...parsed]
        } catch {}
      }
      if (storedTeamRaw && combined.length === 0) {
        try {
          const parsed = JSON.parse(storedTeamRaw)
          if (Array.isArray(parsed)) {
            parsed.forEach((m: any) => {
              combined.push({
                id: m.id,
                userId: m.id.startsWith('WBX') ? m.id : `WBX-EMP-${m.id.substring(0, 4)}`,
                name: m.name,
                email: m.email,
                password: m.portal_password || 'Staff#2026',
                role: m.role_type === 'Team Manager' ? 'Team Manager' : 'Team Member',
                department: m.role_department || m.position || 'Frontend Developer',
                status: m.status || 'active',
                created_at: m.created_at || new Date().toISOString(),
              })
            })
          }
        } catch {}
      }
      setCrmUsers(combined)
    }
    loadUsers()
  }, [])

  const saveCrmUsers = (updated: any[]) => {
    setCrmUsers(updated)
    localStorage.setItem('webotixs_crm_users', JSON.stringify(updated))

    // Also sync team members or client accounts so other pages stay synced
    const teamSync = updated.filter((u) => u.role === 'Team Manager' || u.role === 'Team Member')
    const clientSync = updated.filter((u) => u.role === 'Client')

    if (teamSync.length > 0) {
      localStorage.setItem(
        'webotixs_team_members',
        JSON.stringify(
          teamSync.map((t) => ({
            id: t.userId || t.id,
            name: t.name,
            email: t.email,
            portal_password: t.password,
            role_type: t.role,
            role_department: t.department,
            position: t.department,
            status: t.status,
            created_at: t.created_at,
          }))
        )
      )
    }
    if (clientSync.length > 0) {
      localStorage.setItem(
        'webotixs_client_accounts',
        JSON.stringify(
          clientSync.map((c) => ({
            id: c.userId || c.id,
            clientId: c.userId,
            company_name: c.companyName || c.name,
            contact_name: c.name,
            email: c.email,
            portal_password: c.password,
            phone: c.phone,
            address: c.address,
            website: c.website,
            notes: c.notes,
            status: c.status,
            created_at: c.created_at,
          }))
        )
      )
    }
  }

  const openUserModal = (roleType: 'Team Manager' | 'Team Member' | 'Client', existingUser?: any) => {
    if (existingUser) {
      setEditingUser(existingUser)
      setUserModalType(existingUser.role || roleType)
      setUserForm({
        userId: existingUser.userId || existingUser.id || '',
        name: existingUser.name || '',
        email: existingUser.email || '',
        password: existingUser.password || '',
        department: existingUser.department || 'UI/UX Design',
        companyName: existingUser.companyName || existingUser.name || '',
        phone: existingUser.phone || '',
        address: existingUser.address || '',
        website: existingUser.website || '',
        notes: existingUser.notes || '',
        status: existingUser.status || 'active',
      })
    } else {
      setEditingUser(null)
      setUserModalType(roleType)
      const prefix = roleType === 'Team Manager' ? 'WBX-MGR-' : roleType === 'Team Member' ? 'WBX-EMP-' : 'WBX-CLI-'
      const randomNum = Math.floor(100 + Math.random() * 900)
      const generatedId = `${prefix}${randomNum}`
      const generatedPass = `${roleType === 'Client' ? 'Client' : 'Staff'}#${Math.floor(1000 + Math.random() * 9000)}`

      setUserForm({
        userId: generatedId,
        name: '',
        email: '',
        password: generatedPass,
        department: 'UI/UX Design',
        companyName: '',
        phone: '',
        address: '',
        website: '',
        notes: '',
        status: 'active',
      })
    }
    setIsUserModalOpen(true)
  }

  const handleSaveUserSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!userForm.name || !userForm.email || !userForm.userId) {
      alert('Please fill out Name, User ID, and Email.')
      return
    }

    let updatedList: any[] = []
    if (editingUser) {
      updatedList = crmUsers.map((u) =>
        u.userId === editingUser.userId || u.id === editingUser.id
          ? {
              ...u,
              userId: userForm.userId,
              name: userForm.name,
              email: userForm.email,
              password: userForm.password,
              role: userModalType,
              department: userForm.department,
              companyName: userForm.companyName || userForm.name,
              phone: userForm.phone,
              address: userForm.address,
              website: userForm.website,
              notes: userForm.notes,
              status: userForm.status,
            }
          : u
      )
    } else {
      const newUser = {
        id: crypto.randomUUID(),
        userId: userForm.userId,
        name: userForm.name,
        email: userForm.email,
        password: userForm.password,
        role: userModalType,
        department: userForm.department,
        companyName: userForm.companyName || userForm.name,
        phone: userForm.phone,
        address: userForm.address,
        website: userForm.website,
        notes: userForm.notes,
        status: userForm.status,
        created_at: new Date().toISOString(),
      }
      updatedList = [newUser, ...crmUsers]
    }

    saveCrmUsers(updatedList)
    setIsUserModalOpen(false)
    alert(`✅ ${userModalType} Credentials Saved! The user can now sign into their exact board at /admin/login using ID/email and generated password.`)
  }

  const handleDeleteUser = (user: any) => {
    if (!window.confirm(`Delete account for ${user.name} (${user.userId}) and revoke login access?`)) return
    const updated = crmUsers.filter((u) => u.userId !== user.userId && u.id !== user.id)
    saveCrmUsers(updated)
  }

  const handleToggleUserStatus = (user: any) => {
    const nextStatus = user.status === 'active' ? 'suspended' : 'active'
    const updated = crmUsers.map((u) =>
      u.userId === user.userId || u.id === user.id ? { ...u, status: nextStatus } : u
    )
    saveCrmUsers(updated)
  }

  const handleResetPassword = (user: any) => {
    const newPass = `${user.role === 'Client' ? 'Client' : 'Staff'}#${Math.floor(1000 + Math.random() * 9000)}`
    const updated = crmUsers.map((u) =>
      u.userId === user.userId || u.id === user.id ? { ...u, password: newPass } : u
    )
    saveCrmUsers(updated)
    alert(`🔐 Password reset for ${user.name} (${user.userId})!\nNew Password: ${newPass}`)
  }

  const handleCopyUserCreds = (u: any) => {
    const text = `Webotixs Enterprise Portal Credentials\nLogin Portal: https://webotixs-website.vercel.app/admin/login\nUser ID: ${u.userId}\nEmail: ${u.email}\nPassword: ${u.password}\nAssigned Role: ${u.role}\nDashboard Target: ${u.role === 'Team Manager' ? '/manager/dashboard' : u.role === 'Team Member' ? '/team/dashboard' : '/client/dashboard'}`
    navigator.clipboard.writeText(text)
    setCopiedId(u.userId || u.id)
    setTimeout(() => setCopiedId(null), 2500)
  }

  // Form states for New Project Modal
  const [formTitle, setFormTitle] = useState('')
  const [formCompany, setFormCompany] = useState('')
  const [formClientName, setFormClientName] = useState('')
  const [formEmail, setFormEmail] = useState('')
  const [formPackage, setFormPackage] = useState('Enterprise E-Commerce')
  const [formBudget, setFormBudget] = useState('25000')
  const [formDeadline, setFormDeadline] = useState(new Date(Date.now() + 30 * 24 * 3600000).toISOString().split('T')[0])
  const [formPriority, setFormPriority] = useState('high')
  const [formTemplateId, setFormTemplateId] = useState('33333333-3333-3333-3333-333333333301')
  const [formRequirements, setFormRequirements] = useState('')

  const handleCreateProjectSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await onCreateProject({
        title: formTitle,
        companyName: formCompany,
        clientName: formClientName || formCompany,
        email: formEmail,
        packageType: formPackage,
        budget: formBudget,
        deadline: formDeadline,
        priority: formPriority,
        workflowTemplateId: formTemplateId,
        requirements: formRequirements,
      })
      setIsModalOpen(false)
      setFormTitle('')
      setFormCompany('')
      setFormEmail('')
    } finally {
      setSubmitting(false)
    }
  }

  const filteredProjects = projects.filter((p) =>
    p.title.toLowerCase().includes(search.toLowerCase()) ||
    p.client?.company_name?.toLowerCase().includes(search.toLowerCase())
  )

  const filteredUsers = crmUsers.filter(
    (u) =>
      (userRoleFilter === 'ALL' || u.role === userRoleFilter) &&
      (u.name.toLowerCase().includes(userSearch.toLowerCase()) ||
        u.email.toLowerCase().includes(userSearch.toLowerCase()) ||
        u.userId.toLowerCase().includes(userSearch.toLowerCase()))
  )

  const templatesList = [
    { id: '33333333-3333-3333-3333-333333333301', name: 'WordPress Website', steps: ['UI/UX Design', 'Development', 'SEO Optimization', 'QA Testing', 'Delivery & Client Handoff'], role: 'Full Stack / Agency' },
    { id: '33333333-3333-3333-3333-333333333302', name: 'Shopify Store', steps: ['Store Strategy & Wireframes', 'Shopify Custom Development', 'Product Import & SEO', 'QA & Checkout Testing', 'Launch & Store Delivery'], role: 'E-Commerce' },
    { id: '33333333-3333-3333-3333-333333333303', name: 'Webflow Website', steps: ['UI/UX & Animation Mapping', 'Webflow Build & CMS Setup', 'Technical SEO Audit', 'QA & Responsiveness Check', 'Client Transfer & Delivery'], role: 'Webflow Team' },
    { id: '33333333-3333-3333-3333-333333333304', name: 'UI/UX Design Only', steps: ['UX Research & Persona Mapping', 'Low-Fidelity Wireframing', 'High-Fidelity UI Design', 'Interactive Prototyping', 'Design System Handoff'], role: 'Design Studio' },
    { id: '33333333-3333-3333-3333-333333333305', name: 'SEO Project', steps: ['Comprehensive Technical Audit', 'Keyword & Competitor Matrix', 'On-Page Optimization', 'Content Strategy Execution', 'Analytics Dashboard & Reporting'], role: 'Growth Team' },
    { id: '33333333-3333-3333-3333-333333333306', name: 'AI Automation Project', steps: ['Workflow Mapping & Scoping', 'AI Model Prompt Engineering', 'API & Pipeline Integration', 'Edge Case & Security QA', 'Deployment & Team Training'], role: 'AI Engineering' },
    { id: '33333333-3333-3333-3333-333333333307', name: 'Branding & Graphic Design', steps: ['Brand Discovery & Moodboards', 'Logo Exploration & Concepts', 'Typography & Color System', 'Comprehensive Brand Guidelines', 'Final Asset Kit Delivery'], role: 'Brand Studio' },
  ]

  return (
    <div className="space-y-6">
      {/* Top Nav Tabs */}
      <div className="flex items-center justify-between border-b border-[#273449] pb-3 overflow-x-auto gap-2">
        <div className="flex items-center gap-2">
          {[
            { id: 'users', label: 'Users & Credentials Manager', icon: Users, count: crmUsers.length },
            { id: 'projects', label: 'Projects Hub', icon: Briefcase, count: projects.length },
            { id: 'kanban', label: 'Kanban Board', icon: FolderKanban, count: tasks.length },
            { id: 'calendar', label: 'Calendar & Deadlines', icon: Clock },
            { id: 'team', label: 'Department Staff Overview', icon: Shield, count: crmUsers.filter(u=>u.role!=='Client').length },
            { id: 'templates', label: 'Workflow Templates', icon: Sparkles, count: 7 },
            { id: 'invoices', label: 'Invoices & Payments', icon: Receipt, count: invoices.length },
            { id: 'tickets', label: 'Support Tickets', icon: LifeBuoy, count: tickets.length },
            { id: 'reports', label: 'Agency Reports', icon: BarChart3 },
          ].map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  'flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap',
                  isActive
                    ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-glow-sm'
                    : 'bg-[#0D1224] border border-[#273449] text-[#94A3B8] hover:text-white hover:border-[#273449]/80'
                )}
              >
                <Icon size={14} />
                {tab.label}
                {tab.count !== undefined && (
                  <span className={cn('px-1.5 py-0.2 rounded-full text-[10px] font-bold', isActive ? 'bg-black/20 text-white' : 'bg-[#050816] text-[#94A3B8]')}>
                    {tab.count}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {activeTab === 'projects' && (
          <button
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs rounded-xl shadow-glow-sm transition-all shrink-0"
          >
            <Plus size={16} />
            New Project & Auto-Workflow
          </button>
        )}
      </div>

      {/* TAB 1: USER & CREDENTIALS MANAGER (RBAC GENERATOR) */}
      {activeTab === 'users' && (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-900/30 via-[#0D1224] to-blue-900/30 border border-purple-500/30 rounded-3xl p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-xl">
            <div>
              <span className="px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 text-xs font-bold uppercase tracking-wider">
                Super Admin Command
              </span>
              <h2 className="font-display text-2xl font-bold text-white tracking-tight mt-2 flex items-center gap-2">
                <Key className="text-purple-400" size={22} /> Role-Based User & Credential Generator
              </h2>
              <p className="text-xs text-[#94A3B8] max-w-2xl mt-1">
                Create Team Managers, Team Members, and Clients. Automatically generates User IDs (WBX-EMP-001, WBX-MGR-001, WBX-CLI-001) and temporary passwords for strict role-protected logins.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 shrink-0">
              <button
                onClick={() => openUserModal('Team Manager')}
                className="flex items-center gap-1.5 px-4 py-2.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold rounded-xl transition-all shadow-lg shadow-purple-500/20"
              >
                <Plus size={14} /> + Team Manager
              </button>
              <button
                onClick={() => openUserModal('Team Member')}
                className="flex items-center gap-1.5 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold rounded-xl transition-all shadow-lg shadow-blue-500/20"
              >
                <Plus size={14} /> + Team Member
              </button>
              <button
                onClick={() => openUserModal('Client')}
                className="flex items-center gap-1.5 px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded-xl transition-all shadow-lg shadow-cyan-500/20"
              >
                <Plus size={14} /> + Client Account
              </button>
            </div>
          </div>

          {/* Filter Bar */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-[#0D1224] p-4 rounded-2xl border border-[#273449]">
            <div className="relative w-full sm:max-w-md">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]" />
              <input
                type="text"
                placeholder="Search users by name, User ID, or email..."
                value={userSearch}
                onChange={(e) => setUserSearch(e.target.value)}
                className="w-full pl-11 pr-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/40 focus:outline-none focus:border-purple-500"
              />
            </div>

            <div className="flex items-center gap-2 overflow-x-auto w-full sm:w-auto">
              <span className="text-xs text-[#94A3B8] font-medium shrink-0">Filter Role:</span>
              {(['ALL', 'Team Manager', 'Team Member', 'Client'] as const).map((r) => (
                <button
                  key={r}
                  onClick={() => setUserRoleFilter(r)}
                  className={cn(
                    'px-3 py-1.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap border',
                    userRoleFilter === r
                      ? 'bg-purple-600 text-white border-purple-500 shadow'
                      : 'bg-[#050816] text-[#94A3B8] border-[#273449] hover:text-white'
                  )}
                >
                  {r === 'ALL' ? 'All Roles' : r}
                </button>
              ))}
            </div>
          </div>

          {/* Users Table */}
          <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[#273449] text-[#94A3B8] uppercase tracking-wider text-left bg-[#050816]">
                    <th className="py-3.5 px-4 font-semibold">User ID</th>
                    <th className="py-3.5 px-4 font-semibold">Name & Email</th>
                    <th className="py-3.5 px-4 font-semibold">Role / Department</th>
                    <th className="py-3.5 px-4 font-semibold">Login Password</th>
                    <th className="py-3.5 px-4 font-semibold text-center">Status</th>
                    <th className="py-3.5 px-4 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredUsers.map((u) => (
                    <tr key={u.id || u.userId} className="border-b border-[#273449]/60 hover:bg-white/[0.02] transition-colors">
                      <td className="py-4 px-4 font-mono font-bold text-purple-400">{u.userId}</td>
                      <td className="py-4 px-4">
                        <div className="font-bold text-white text-sm">{u.name}</div>
                        <div className="text-[11px] text-[#94A3B8] flex items-center gap-1 mt-0.5 font-mono">
                          <Mail size={11} className="text-blue-400" /> {u.email}
                        </div>
                      </td>
                      <td className="py-4 px-4">
                        <span className={cn(
                          'px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase border',
                          u.role === 'Team Manager'
                            ? 'bg-purple-500/10 text-purple-400 border-purple-500/30'
                            : u.role === 'Client'
                            ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30'
                            : 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                        )}>
                          {u.role}
                        </span>
                        {u.department && <div className="text-[11px] text-[#94A3B8] mt-1">{u.department}</div>}
                      </td>
                      <td className="py-4 px-4 font-mono font-bold text-emerald-400 tracking-wider">
                        {u.password || '••••••••'}
                      </td>
                      <td className="py-4 px-4 text-center">
                        <span className={cn(
                          'px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase border',
                          u.status === 'active'
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                            : 'bg-red-500/10 text-red-400 border-red-500/30'
                        )}>
                          {u.status}
                        </span>
                      </td>
                      <td className="py-4 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => handleCopyUserCreds(u)}
                            className="p-1.5 rounded-lg bg-[#050816] border border-[#273449] hover:border-purple-500 text-purple-400 transition-colors"
                            title="Copy Portal Login Credentials"
                          >
                            {copiedId === (u.userId || u.id) ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
                          </button>
                          <button
                            onClick={() => openUserModal(u.role, u)}
                            className="px-2.5 py-1 rounded-lg bg-[#050816] border border-[#273449] hover:border-blue-500 text-blue-400 font-semibold transition-colors text-[11px]"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => handleResetPassword(u)}
                            className="p-1.5 rounded-lg bg-[#050816] border border-[#273449] hover:border-amber-500 text-amber-400 transition-colors"
                            title="Reset Password"
                          >
                            <RefreshCw size={13} />
                          </button>
                          <button
                            onClick={() => handleToggleUserStatus(u)}
                            className={cn('p-1.5 rounded-lg bg-[#050816] border border-[#273449] transition-colors', u.status === 'active' ? 'hover:border-red-500 text-red-400' : 'hover:border-emerald-500 text-emerald-400')}
                            title={u.status === 'active' ? 'Suspend Access' : 'Activate Access'}
                          >
                            {u.status === 'active' ? <UserX size={14} /> : <UserCheck size={14} />}
                          </button>
                          <button
                            onClick={() => handleDeleteUser(u)}
                            className="p-1.5 rounded-lg bg-[#050816] border border-[#273449] hover:border-red-500 text-red-400 transition-colors"
                            title="Delete User Account"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {filteredUsers.length === 0 && (
                    <tr>
                      <td colSpan={6} className="py-12 text-center text-[#94A3B8] bg-[#050816]/50">
                        <Users size={32} className="mx-auto mb-2 opacity-60 text-purple-400" />
                        <div className="font-bold text-white text-sm">No Users Found</div>
                        <p className="text-xs mt-1">Click the + buttons above to create your first Team Manager, Team Member, or Client account.</p>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: PROJECTS HUB */}
      {activeTab === 'projects' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between gap-4">
            <div className="relative max-w-sm w-full">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]" />
              <input
                type="text"
                placeholder="Search projects or clients..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-xs focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProjects.map((project) => (
              <div
                key={project.id}
                onClick={() => onOpenProject(project)}
                className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 space-y-4 hover:border-blue-500/50 transition-all cursor-pointer shadow-lg group"
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase tracking-wider">
                    {project.package_type || 'Custom Workflow'}
                  </span>
                  <span className={cn('px-2.5 py-1 rounded-full text-[10px] font-bold uppercase border', project.status === 'Completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-blue-500/10 text-blue-400 border-blue-500/30')}>
                    {project.status || 'In Progress'}
                  </span>
                </div>

                <div>
                  <h3 className="font-display font-bold text-white text-base group-hover:text-blue-400 transition-colors line-clamp-1">{project.title}</h3>
                  <p className="text-xs text-[#94A3B8] font-medium mt-1">{project.client?.company_name || 'Assigned Client'}</p>
                </div>

                <div className="space-y-1.5 pt-2 border-t border-[#273449]/50">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#94A3B8]">Completion</span>
                    <span className="text-white font-bold">{project.progress_percentage || 0}%</span>
                  </div>
                  <div className="h-2 w-full bg-[#050816] rounded-full overflow-hidden border border-[#273449]">
                    <div className="h-full bg-gradient-to-r from-blue-600 to-cyan-500 rounded-full transition-all duration-500" style={{ width: `${project.progress_percentage || 0}%` }} />
                  </div>
                </div>

                <div className="flex items-center justify-between text-[11px] text-[#94A3B8] pt-1">
                  <span>Deadline: <strong className="text-white">{project.deadline || 'TBD'}</strong></span>
                  <span>Budget: <strong className="text-emerald-400 font-mono">${project.budget?.toLocaleString() || '25,000'}</strong></span>
                </div>
              </div>
            ))}

            {filteredProjects.length === 0 && (
              <div className="col-span-full bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
                <Briefcase size={36} className="text-blue-400 mx-auto opacity-70" />
                <h3 className="font-display font-bold text-white text-base">No Projects Created Yet</h3>
                <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                  Click &apos;New Project & Auto-Workflow&apos; above to initialize your first project, generate a client login account, and spawn lifecycle tasks.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: KANBAN BOARD */}
      {activeTab === 'kanban' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Agency Lifecycle Task Kanban</h3>
            <p className="text-[#94A3B8] text-xs">Drag and drop or update step tasks across Todo, In Progress, Review, and Completed stages.</p>
          </div>
          <KanbanBoard tasks={tasks} onStatusChange={onTaskStatusChange} onTaskClick={(t) => {
            const p = projects.find(proj => proj.id === t.project_id)
            if (p) onOpenProject(p)
          }} />
        </div>
      )}

      {/* TAB 4: CALENDAR & DEADLINES */}
      {activeTab === 'calendar' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl">
          <CalendarView projects={projects} tasks={tasks} onTaskClick={(t) => {
            const p = projects.find(proj => proj.id === t.project_id)
            if (p) onOpenProject(p)
          }} />
        </div>
      )}

      {/* TAB 5: DEPARTMENT STAFF OVERVIEW */}
      {activeTab === 'team' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {crmUsers.filter(u => u.role !== 'Client').map((member) => (
              <div key={member.id || member.userId} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 space-y-4 shadow-xl">
                <div className="flex items-center gap-3.5">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-600/20 to-blue-600/20 border border-purple-500/30 flex items-center justify-center font-display font-bold text-lg text-purple-400">
                    {member.name ? member.name.split(' ').map((n: any) => n[0]).join('') : 'TM'}
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-white text-base">{member.name}</h4>
                    <span className="text-xs text-purple-400 font-semibold">{member.role} — {member.department}</span>
                    <div className="text-[11px] text-[#94A3B8] font-mono mt-0.5">{member.email}</div>
                  </div>
                </div>

                <div className="pt-3 border-t border-[#273449] flex items-center justify-between text-xs">
                  <span className="text-[#94A3B8]">ID: <strong className="text-white font-mono">{member.userId}</strong></span>
                  <span className={cn('px-2 py-0.5 rounded text-[10px] font-bold uppercase border', member.status === 'active' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-red-500/10 text-red-400 border-red-500/30')}>
                    {member.status}
                  </span>
                </div>
              </div>
            ))}
            {crmUsers.filter(u => u.role !== 'Client').length === 0 && (
              <div className="col-span-full bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
                <Shield size={36} className="text-purple-400 mx-auto opacity-70" />
                <h3 className="font-display font-bold text-white text-base">No Department Staff Created</h3>
                <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                  Use the Users & Credentials Manager tab right above to generate accounts for your Team Managers and Team Members.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 6: WORKFLOW TEMPLATES */}
      {activeTab === 'templates' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-6">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Preset Agency Workflow Templates (Auto-Generated on Project Creation)</h3>
            <p className="text-[#94A3B8] text-xs mt-1">
              When you select a template during project creation, all sequential tasks are automatically generated, assigned to appropriate roles, and linked with dependency rules.
            </p>
          </div>

          <div className="space-y-4">
            {templatesList.map((tpl) => (
              <div key={tpl.id} className="p-5 rounded-2xl bg-[#050816] border border-[#273449] space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#273449] pb-3">
                  <div>
                    <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">{tpl.role}</span>
                    <h4 className="font-display font-bold text-white text-base mt-0.5">{tpl.name}</h4>
                  </div>
                  <span className="text-xs text-[#94A3B8] font-semibold">5 Sequential Lifecycle Steps</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
                  {tpl.steps.map((stepName, idx) => (
                    <div key={idx} className="p-3 rounded-xl bg-[#0D1224] border border-[#273449]/80 space-y-1 relative">
                      <div className="text-[10px] font-bold text-blue-400 uppercase">Step #{idx + 1}</div>
                      <div className="text-xs font-semibold text-white truncate" title={stepName}>{stepName}</div>
                      <div className="text-[10px] text-[#94A3B8] flex items-center gap-1 mt-1">
                        {idx === 0 ? <CheckCircle2 size={11} className="text-emerald-400" /> : <Clock size={11} className="text-amber-400" />}
                        {idx === 0 ? 'Unlocked initially' : `Unlocks after Step #${idx}`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 7: INVOICES */}
      {activeTab === 'invoices' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Client Invoices & Milestone Payments</h3>
            <p className="text-[#94A3B8] text-xs">Automated billing connected to client portals with PDF download readiness.</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#273449] text-[#94A3B8] uppercase tracking-wider text-left">
                  <th className="py-3 px-4 font-semibold">Invoice #</th>
                  <th className="py-3 px-4 font-semibold">Client Company</th>
                  <th className="py-3 px-4 font-semibold">Project</th>
                  <th className="py-3 px-4 font-semibold">Amount</th>
                  <th className="py-3 px-4 font-semibold">Due Date</th>
                  <th className="py-3 px-4 font-semibold text-center">Status</th>
                  <th className="py-3 px-4 font-semibold text-right">PDF Action</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((inv) => (
                  <tr key={inv.id} className="border-b border-[#273449]/60 hover:bg-white/[0.02] transition-colors">
                    <td className="py-3.5 px-4 font-mono font-bold text-white">{inv.invoice_number}</td>
                    <td className="py-3.5 px-4 font-semibold text-white">{inv.client?.company_name || 'Client Company'}</td>
                    <td className="py-3.5 px-4 text-[#94A3B8]">{inv.project?.title || 'Project'}</td>
                    <td className="py-3.5 px-4 font-bold text-emerald-400">${inv.total_amount?.toLocaleString() || '0.00'}</td>
                    <td className="py-3.5 px-4 text-[#94A3B8] font-mono">{inv.due_date}</td>
                    <td className="py-3.5 px-4 text-center">
                      <span className={cn(
                        'px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase border',
                        inv.status === 'Paid' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      )}>
                        {inv.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => alert(`Downloading PDF statement for ${inv.invoice_number}...`)}
                        className="p-2 rounded-lg bg-[#050816] border border-[#273449] hover:border-blue-500 text-blue-400 transition-colors inline-flex items-center gap-1 text-[11px] font-bold"
                      >
                        <Download size={13} /> PDF
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {invoices.length === 0 && (
              <div className="py-12 text-center text-[#94A3B8]">No invoices created yet.</div>
            )}
          </div>
        </div>
      )}

      {/* TAB 8: SUPPORT TICKETS */}
      {activeTab === 'tickets' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 shadow-xl space-y-4">
          <div>
            <h3 className="font-display font-bold text-white text-lg">Client Support Tickets Hub</h3>
            <p className="text-[#94A3B8] text-xs">Direct support pipeline initiated by clients inside their portal.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {tickets.map((t) => (
              <div key={t.id} className="p-4 rounded-xl bg-[#050816] border border-[#273449] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-blue-400">{t.ticket_number} • {t.priority} Priority</span>
                  <span className={cn(
                    'px-2 py-0.5 rounded text-[10px] font-bold uppercase border',
                    t.status === 'Open' || t.status === 'Working' ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  )}>
                    {t.status}
                  </span>
                </div>
                <h4 className="font-bold text-white text-sm">{t.subject}</h4>
                <p className="text-[#94A3B8] text-xs leading-relaxed">{t.description}</p>
                <div className="pt-2 border-t border-[#273449]/50 flex items-center justify-between text-[11px] text-[#94A3B8]">
                  <span>Client: {t.client?.company_name || 'Client'}</span>
                  <span>📅 {formatDate(t.created_at || new Date().toISOString())}</span>
                </div>
              </div>
            ))}
            {tickets.length === 0 && (
              <div className="col-span-full py-12 text-center text-[#94A3B8]">No open support tickets.</div>
            )}
          </div>
        </div>
      )}

      {/* TAB 9: REPORTS & ANALYTICS */}
      {activeTab === 'reports' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { title: 'Total Active Projects', value: projects.length.toString(), sub: 'SaaS pipeline tracking', color: 'text-blue-400' },
              { title: 'Total Users & Staff', value: crmUsers.length.toString(), sub: 'Across 4 RBAC roles', color: 'text-purple-400' },
              { title: 'Total Pipeline Tasks', value: tasks.length.toString(), sub: 'Automated templates', color: 'text-cyan-400' },
              { title: 'Invoices Issued', value: invoices.length.toString(), sub: 'Billing connected', color: 'text-emerald-400' },
            ].map((metric, i) => (
              <div key={i} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-5 shadow-xl space-y-1">
                <span className="text-xs font-bold text-[#94A3B8] uppercase tracking-wider">{metric.title}</span>
                <div className={cn('font-display font-bold text-2xl', metric.color)}>{metric.value}</div>
                <div className="text-[11px] text-[#94A3B8]/80">{metric.sub}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* MODAL: CREATE / EDIT USER CREDENTIALS */}
      {isUserModalOpen && (
        <div className="fixed inset-0 z-[120] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto">
          <div className="w-full max-w-xl bg-[#0D1224] border border-[#273449] rounded-3xl p-6 shadow-2xl space-y-5 my-8">
            <div className="flex items-center justify-between border-b border-[#273449] pb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center text-purple-400">
                  <Key size={18} />
                </div>
                <div>
                  <h3 className="font-display font-bold text-white text-lg">
                    {editingUser ? 'Edit User Portal Credentials' : `Create New ${userModalType} Account`}
                  </h3>
                  <p className="text-[#94A3B8] text-xs">Generates unique User ID & temporary login password for route protection.</p>
                </div>
              </div>
              <button onClick={() => setIsUserModalOpen(false)} className="p-2 rounded-xl hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSaveUserSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white flex items-center gap-1.5">
                    User ID * (Auto-Generated)
                  </label>
                  <input
                    type="text"
                    required
                    value={userForm.userId}
                    onChange={(e) => setUserForm({ ...userForm, userId: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-purple-500/40 rounded-xl text-purple-300 font-mono font-bold text-xs focus:border-purple-500"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Full Name / Company Name *</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Sarah Chen or Al-Khaleej Group"
                    value={userForm.name}
                    onChange={(e) => setUserForm({ ...userForm, name: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white flex items-center gap-1.5">
                    <Mail size={12} className="text-blue-400" /> Login Email / Username *
                  </label>
                  <input
                    type="email"
                    required
                    placeholder="user@webotixs.com"
                    value={userForm.email}
                    onChange={(e) => setUserForm({ ...userForm, email: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50 font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white flex items-center justify-between">
                    <span className="flex items-center gap-1.5"><Lock size={12} className="text-emerald-400" /> Portal Password *</span>
                    <button
                      type="button"
                      onClick={() => setUserForm({ ...userForm, password: `${userModalType === 'Client' ? 'Client' : 'Staff'}#${Math.floor(1000 + Math.random() * 9000)}` })}
                      className="text-[10px] text-cyan-400 hover:underline"
                    >
                      Regenerate
                    </button>
                  </label>
                  <input
                    type="text"
                    required
                    value={userForm.password}
                    onChange={(e) => setUserForm({ ...userForm, password: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-emerald-500/40 rounded-xl text-emerald-400 font-mono font-bold text-xs focus:border-emerald-500"
                  />
                </div>
              </div>

              {userModalType !== 'Client' && (
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Assigned Department / Role Board *</label>
                  <select
                    value={userForm.department}
                    onChange={(e) => setUserForm({ ...userForm, department: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-purple-500"
                  >
                    <option value="UI/UX Designer">UI/UX Designer</option>
                    <option value="Frontend Developer">Frontend Developer</option>
                    <option value="Backend Developer">Backend Developer</option>
                    <option value="WordPress Developer">WordPress Developer</option>
                    <option value="QA Tester">QA Tester</option>
                    <option value="SEO Specialist">SEO Specialist</option>
                    <option value="Project Manager">Project Manager</option>
                  </select>
                </div>
              )}

              {userModalType === 'Client' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-xs font-bold text-white">Contact Phone</label>
                    <input
                      type="text"
                      placeholder="+971 50 123 4567"
                      value={userForm.phone}
                      onChange={(e) => setUserForm({ ...userForm, phone: e.target.value })}
                      className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-bold text-white">Company Website</label>
                    <input
                      type="text"
                      placeholder="https://company.com"
                      value={userForm.website}
                      onChange={(e) => setUserForm({ ...userForm, website: e.target.value })}
                      className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                    />
                  </div>
                </div>
              )}

              <div className="space-y-1">
                <label className="text-xs font-bold text-white">Internal Notes</label>
                <textarea
                  rows={2}
                  placeholder="Special instructions, permissions, notes..."
                  value={userForm.notes}
                  onChange={(e) => setUserForm({ ...userForm, notes: e.target.value })}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs resize-none"
                />
              </div>

              <div className="p-3.5 rounded-xl bg-purple-600/10 border border-purple-500/30 text-xs text-purple-300 flex items-center gap-3">
                <Shield size={18} className="shrink-0 text-purple-400" />
                <span>
                  <strong>Strict RBAC Enforced:</strong> When signed in with these credentials, the user will be routed directly and exclusively to their assigned {userModalType === 'Team Manager' ? '/manager/dashboard' : userModalType === 'Team Member' ? '/team/dashboard' : '/client/dashboard'}.
                </span>
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsUserModalOpen(false)}
                  className="px-5 py-2.5 rounded-xl bg-[#050816] border border-[#273449] text-xs font-bold text-[#94A3B8] hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-bold text-xs shadow-glow-sm transition-all"
                >
                  {editingUser ? 'Save Credential Changes' : `Generate ${userModalType} Account`}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* NEW PROJECT MODAL */}
      {isModalOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto">
          <div className="w-full max-w-2xl bg-[#0D1224] border border-[#273449] rounded-3xl p-6 shadow-2xl space-y-5 my-8">
            <div className="flex items-center justify-between border-b border-[#273449] pb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-blue-600/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
                  <Sparkles size={18} />
                </div>
                <div>
                  <h3 className="font-display font-bold text-white text-lg">Create New Project & Auto-Generate Workflow</h3>
                  <p className="text-[#94A3B8] text-xs">Automatically creates Client account, generates secure credentials, and spawns template tasks.</p>
                </div>
              </div>
              <button onClick={() => setIsModalOpen(false)} className="p-2 rounded-xl hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleCreateProjectSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Project Title *</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Al-Khaleej E-Commerce Overhaul"
                    value={formTitle}
                    onChange={(e) => setFormTitle(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Client Company Name *</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Al-Khaleej Group"
                    value={formCompany}
                    onChange={(e) => setFormCompany(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Client Email * (For Portal Credentials)</label>
                  <input
                    type="email"
                    required
                    placeholder="client@company.com"
                    value={formEmail}
                    onChange={(e) => setFormEmail(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Workflow Template * (Auto-Generates Tasks)</label>
                  <select
                    value={formTemplateId}
                    onChange={(e) => setFormTemplateId(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                  >
                    {templatesList.map((tpl) => (
                      <option key={tpl.id} value={tpl.id}>
                        {tpl.name} ({tpl.steps.length} Steps)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Package Type</label>
                  <input
                    type="text"
                    value={formPackage}
                    onChange={(e) => setFormPackage(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Budget ($)</label>
                  <input
                    type="number"
                    value={formBudget}
                    onChange={(e) => setFormBudget(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-bold text-white">Deadline</label>
                  <input
                    type="date"
                    value={formDeadline}
                    onChange={(e) => setFormDeadline(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-white">Project Requirements & Notes</label>
                <textarea
                  rows={3}
                  placeholder="Specific client notes, RTL requirements, API endpoints..."
                  value={formRequirements}
                  onChange={(e) => setFormRequirements(e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:border-blue-500/50"
                />
              </div>

              <div className="p-3.5 rounded-xl bg-blue-600/10 border border-blue-500/30 flex items-center gap-3 text-xs text-blue-300">
                <CheckCircle2 size={18} className="shrink-0 text-blue-400" />
                <span>
                  <strong>Automatic Client Account:</strong> Generates unique Client ID, temporary password, secret token, and dispatches portal login credentials via email immediately.
                </span>
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-5 py-2.5 rounded-xl bg-[#050816] border border-[#273449] text-xs font-bold text-[#94A3B8] hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs shadow-glow-sm transition-all"
                >
                  {submitting ? 'Creating Project & Workflow...' : 'Create Project & Auto-Assign Team'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
