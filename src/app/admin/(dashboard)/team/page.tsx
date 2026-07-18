'use client'

import { useState, useEffect } from 'react'
import { Plus, Pencil, Trash2, Search, X, Save, Loader2, Linkedin, Github, Mail, Key, Shield, Copy, Check } from 'lucide-react'
import ImagePicker from '@/components/admin/ImagePicker'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'
import type { TeamMember } from '@/lib/types'
import { cn } from '@/lib/utils'

const departmentRoles = [
  'UI/UX Designer',
  'Frontend Developer',
  'Backend Developer',
  'WordPress Developer',
  'SEO Specialist',
  'Content Writer',
  'QA Tester',
  'Project Manager',
  'Admin',
]

export default function AdminTeamPage() {
  const [members, setMembers] = useState<TeamMember[]>([])
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<Partial<TeamMember> | null>(null)
  const [isNew, setIsNew] = useState(false)
  const [saving, setSaving] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  useEffect(() => {
    async function loadTeam() {
      try {
        const data = await getCMSData<TeamMember[]>('team')
        if (Array.isArray(data) && data.length > 0) {
          setMembers(data)
          localStorage.setItem('webotixs_team_members', JSON.stringify(data))
          return
        }
        const local = localStorage.getItem('webotixs_team_members')
        if (local) {
          const parsed = JSON.parse(local)
          if (Array.isArray(parsed)) setMembers(parsed)
        }
      } catch (e) {
        const local = localStorage.getItem('webotixs_team_members')
        if (local) {
          const parsed = JSON.parse(local)
          if (Array.isArray(parsed)) setMembers(parsed)
        }
      }
    }
    loadTeam()
  }, [])

  const filtered = members.filter(
    (m) =>
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.position.toLowerCase().includes(search.toLowerCase()) ||
      (m.email && m.email.toLowerCase().includes(search.toLowerCase()))
  )

  const openNew = () => {
    const genPass = `Staff#${Math.floor(1000 + Math.random() * 9000)}`
    setEditing({
      name: '',
      position: 'Frontend Developer',
      role_department: 'Frontend Developer',
      bio: '',
      linkedin: '',
      github: '',
      email: '',
      portal_password: genPass,
      status: 'active',
      featured: false,
      order: members.length + 1,
      photo: null,
    })
    setIsNew(true)
  }

  const openEdit = (m: TeamMember) => {
    setEditing({
      ...m,
      role_department: m.role_department || m.position || 'Frontend Developer',
      portal_password: m.portal_password || `Staff#${Math.floor(1000 + Math.random() * 9000)}`,
    })
    setIsNew(false)
  }

  const close = () => {
    setEditing(null)
    setIsNew(false)
  }

  const handleSave = async () => {
    if (!editing?.name || !editing?.position || !editing?.email) {
      alert('Please enter Name, Position, and Login Email.')
      return
    }
    setSaving(true)
    const now = new Date().toISOString()
    let updatedMembers: TeamMember[] = []

    if (isNew) {
      const newMem = {
        ...(editing as TeamMember),
        id: crypto.randomUUID(),
        photo: editing.photo || null,
        created_at: now,
      }
      updatedMembers = [newMem, ...members]
      setMembers(updatedMembers)
      await saveCMSItem('team', newMem)
    } else {
      const updated = { ...editing } as TeamMember
      updatedMembers = members.map((m) => (m.id === editing.id ? updated : m))
      setMembers(updatedMembers)
      await saveCMSItem('team', updated)
    }

    localStorage.setItem('webotixs_team_members', JSON.stringify(updatedMembers))
    setSaving(false)
    close()
    alert(`✅ Team Member Portal Credentials Saved! They can now log in at /admin/login using their email & generated password.`)
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete team member & revoke portal login access?')) return
    const updated = members.filter((m) => m.id !== id)
    setMembers(updated)
    localStorage.setItem('webotixs_team_members', JSON.stringify(updated))
    await deleteCMSItem('team', id)
  }

  const toggleStatus = async (id: string) => {
    const target = members.find((m) => m.id === id)
    if (!target) return
    const updated = { ...target, status: target.status === 'active' ? 'inactive' : 'active' } as TeamMember
    const updatedMembers = members.map((m) => (m.id === id ? updated : m))
    setMembers(updatedMembers)
    localStorage.setItem('webotixs_team_members', JSON.stringify(updatedMembers))
    await saveCMSItem('team', updated)
  }

  const handleCopyCredentials = (m: TeamMember) => {
    const text = `Webotixs Agency — Staff Portal Login\nLogin URL: https://webotixs-website.vercel.app/admin/login\nEmail / User ID: ${m.email || 'N/A'}\nPassword: ${m.portal_password || 'N/A'}\nAssigned Board: ${m.role_department || m.position}`
    navigator.clipboard.writeText(text)
    setCopiedId(m.id)
    setTimeout(() => setCopiedId(null), 2500)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Team Portal ID Generator & Manager</h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Add staff members, assign department roles, and generate unique portal credentials so they can access their specific workspace boards.
          </p>
        </div>
        <button
          onClick={openNew}
          className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-purple-600 to-blue-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all shadow-lg shadow-purple-500/20"
        >
          <Plus size={16} /> Add Team Member & Generate ID
        </button>
      </div>

      <div className="relative max-w-md">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search team by name, position, or email..."
          className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-purple-500/50"
        />
      </div>

      {/* Team Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filtered.map((member) => (
          <div
            key={member.id}
            className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 flex flex-col justify-between hover:border-purple-500/40 transition-all shadow-md"
          >
            <div>
              <div className="flex items-start justify-between gap-4 mb-4">
                <div className="flex items-center gap-3.5">
                  {member.photo ? (
                    <img src={member.photo} alt={member.name} className="w-14 h-14 rounded-2xl object-cover border border-purple-500/30 shadow" />
                  ) : (
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-600/20 to-blue-600/20 border border-purple-500/30 flex items-center justify-center font-display font-bold text-lg text-purple-400">
                      {member.name.split(' ').map((n) => n[0]).join('')}
                    </div>
                  )}
                  <div>
                    <h3 className="font-display font-bold text-white text-base">{member.name}</h3>
                    <p className="text-xs text-purple-400 font-semibold">{member.position}</p>
                    <span className="inline-block mt-1 px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-[10px] text-blue-400 font-medium">
                      Board: {member.role_department || member.position}
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => toggleStatus(member.id)}
                  className={cn(
                    'px-2.5 py-1 rounded-full text-[9px] font-bold uppercase tracking-wider border shrink-0',
                    member.status === 'active'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                      : 'bg-red-500/10 text-red-400 border-red-500/30'
                  )}
                >
                  {member.status}
                </button>
              </div>

              {member.bio && <p className="text-xs text-[#94A3B8] line-clamp-2 mb-4">{member.bio}</p>}

              {/* Portal Credentials Box */}
              <div className="bg-[#050816] border border-[#273449] rounded-xl p-3.5 mb-4 space-y-2">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-[#94A3B8] flex items-center gap-1.5 font-medium">
                    <Mail size={12} className="text-blue-400" /> Login User ID:
                  </span>
                  <span className="text-white font-mono font-semibold">{member.email || 'No Email Set'}</span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-[#94A3B8] flex items-center gap-1.5 font-medium">
                    <Key size={12} className="text-purple-400" /> Portal Password:
                  </span>
                  <span className="text-emerald-400 font-mono font-bold tracking-wider">{member.portal_password || 'Staff#2026'}</span>
                </div>
                <button
                  onClick={() => handleCopyCredentials(member)}
                  className="w-full mt-1.5 py-1.5 px-3 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 rounded-lg text-[11px] font-semibold text-purple-300 flex items-center justify-center gap-1.5 transition-all"
                >
                  {copiedId === member.id ? (
                    <>
                      <Check size={13} className="text-emerald-400" /> Copied Portal Login Credentials!
                    </>
                  ) : (
                    <>
                      <Copy size={13} /> Copy Login Details to Hand to Staff
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-[#273449]/50 text-xs">
              <div className="flex items-center gap-1.5">
                {member.linkedin && <a href={member.linkedin} target="_blank" rel="noreferrer" className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-blue-400"><Linkedin size={13} /></a>}
                {member.github && <a href={member.github} target="_blank" rel="noreferrer" className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-purple-400"><Github size={13} /></a>}
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => openEdit(member)}
                  className="px-3 py-1.5 rounded-lg bg-[#050816] border border-[#273449] text-[#94A3B8] hover:text-white font-medium transition-colors flex items-center gap-1"
                >
                  <Pencil size={12} /> Edit / Reset Pass
                </button>
                <button
                  onClick={() => handleDelete(member.id)}
                  className="p-1.5 rounded-lg bg-[#050816] border border-[#273449] text-[#94A3B8] hover:text-red-400 transition-colors"
                  title="Delete Member"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <div className="col-span-full bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center">
            <Shield size={36} className="text-purple-400 mx-auto mb-3 opacity-80" />
            <h3 className="font-display font-bold text-white text-base">No Team Members or Portal IDs Generated Yet</h3>
            <p className="text-[#94A3B8] text-xs max-w-md mx-auto mt-1 mb-5">
              Click the button below to add your first staff member, assign their department board, and generate their portal ID and login password.
            </p>
            <button
              onClick={openNew}
              className="px-5 py-2.5 bg-gradient-to-r from-purple-600 to-blue-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all"
            >
              + Add First Team Member
            </button>
          </div>
        )}
      </div>

      {/* Modal Editor & Generator */}
      {editing && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col shadow-2xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white flex items-center gap-2">
                <Key className="text-purple-400" size={18} />
                {isNew ? 'Generate Staff Portal Account' : 'Edit Staff Member'}
              </h2>
              <button onClick={close} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Full Name *</label>
                  <input
                    type="text"
                    value={editing.name || ''}
                    onChange={(e) => setEditing({ ...editing, name: e.target.value })}
                    placeholder="e.g. Sarah Jenkins"
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Position Title *</label>
                  <input
                    type="text"
                    value={editing.position || ''}
                    onChange={(e) => setEditing({ ...editing, position: e.target.value })}
                    placeholder="e.g. Lead UI/UX Designer"
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500/50"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 bg-purple-500/5 border border-purple-500/20 p-4 rounded-2xl">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-purple-300 flex items-center gap-1.5">
                    <Mail size={13} /> Login User ID (Email) *
                  </label>
                  <input
                    type="email"
                    value={editing.email || ''}
                    onChange={(e) => setEditing({ ...editing, email: e.target.value })}
                    placeholder="staff@webotixs.com"
                    className="w-full px-4 py-2.5 bg-[#050816] border border-purple-500/30 rounded-xl text-white text-sm focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-purple-300 flex items-center justify-between">
                    <span className="flex items-center gap-1.5">
                      <Key size={13} /> Portal Password *
                    </span>
                    <button
                      type="button"
                      onClick={() => setEditing({ ...editing, portal_password: `Staff#${Math.floor(1000 + Math.random() * 9000)}` })}
                      className="text-[10px] text-cyan-400 hover:underline font-bold"
                    >
                      Regenerate
                    </button>
                  </label>
                  <input
                    type="text"
                    value={editing.portal_password || ''}
                    onChange={(e) => setEditing({ ...editing, portal_password: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-purple-500/30 rounded-xl text-emerald-400 font-mono font-bold text-sm focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Assigned Department Board (Role Filter) *</label>
                <select
                  value={editing.role_department || 'Frontend Developer'}
                  onChange={(e) => setEditing({ ...editing, role_department: e.target.value, position: editing.position || e.target.value })}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500/50"
                >
                  {departmentRoles.map((role) => (
                    <option key={role} value={role} className="bg-[#0D1224] text-white">
                      {role}
                    </option>
                  ))}
                </select>
                <p className="text-[11px] text-[#94A3B8]">
                  When this staff member logs in via /admin/login, they will be sent straight to the Team Portal viewing exclusively deliverables requiring this role.
                </p>
              </div>

              <ImagePicker
                label="Staff Profile Photo (Optional)"
                value={editing.photo || null}
                onChange={(url) => setEditing({ ...editing, photo: url })}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Short Bio</label>
                <textarea
                  rows={2}
                  value={editing.bio || ''}
                  onChange={(e) => setEditing({ ...editing, bio: e.target.value })}
                  placeholder="Brief summary of expertise..."
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500/50 resize-none"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">LinkedIn URL</label>
                  <input
                    type="text"
                    value={editing.linkedin || ''}
                    onChange={(e) => setEditing({ ...editing, linkedin: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">GitHub URL</label>
                  <input
                    type="text"
                    value={editing.github || ''}
                    onChange={(e) => setEditing({ ...editing, github: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500/50"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449] bg-[#050816]/50">
              <button onClick={close} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-purple-600 to-blue-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50"
              >
                {saving ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> Saving...
                  </>
                ) : (
                  <>
                    <Save size={14} /> {isNew ? 'Generate & Save Staff ID' : 'Save Changes'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
