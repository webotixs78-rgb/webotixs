'use client'

import { useState, useEffect } from 'react'
import { Plus, Pencil, Trash2, Search, X, Save, Loader2, Linkedin, Github, Mail } from 'lucide-react'
import ImagePicker from '@/components/admin/ImagePicker'
import { mockTeam } from '@/lib/data/mock'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'
import type { TeamMember } from '@/lib/types'
import { cn } from '@/lib/utils'

const emptyMember: Partial<TeamMember> = {
  name: '', position: '', bio: '', linkedin: '', github: '', email: '',
  status: 'active', featured: false, order: 0, photo: null,
}

export default function AdminTeamPage() {
  const [members, setMembers] = useState<TeamMember[]>([...mockTeam])
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<Partial<TeamMember> | null>(null)
  const [isNew, setIsNew] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getCMSData<TeamMember[]>('team').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setMembers(data)
      }
    })
  }, [])

  const filtered = members.filter((m) => m.name.toLowerCase().includes(search.toLowerCase()) || m.position.toLowerCase().includes(search.toLowerCase()))

  const openNew = () => { setEditing({ ...emptyMember }); setIsNew(true) }
  const openEdit = (m: TeamMember) => { setEditing({ ...m }); setIsNew(false) }
  const close = () => { setEditing(null); setIsNew(false) }

  const handleSave = async () => {
    if (!editing?.name || !editing?.position) return
    setSaving(true)
    const now = new Date().toISOString()
    if (isNew) {
      const newMem = { ...(editing as TeamMember), id: crypto.randomUUID(), photo: editing.photo || null, created_at: now }
      setMembers((prev) => [newMem, ...prev])
      await saveCMSItem('team', newMem)
    } else {
      const updated = { ...editing } as TeamMember
      setMembers((prev) => prev.map((m) => m.id === editing.id ? updated : m))
      await saveCMSItem('team', updated)
    }
    setSaving(false)
    close()
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete team member?')) return
    setMembers((prev) => prev.filter((m) => m.id !== id))
    await deleteCMSItem('team', id)
  }

  const toggleStatus = async (id: string) => {
    const target = members.find((m) => m.id === id)
    if (!target) return
    const updated = { ...target, status: target.status === 'active' ? 'inactive' : 'active' } as TeamMember
    setMembers((prev) => prev.map((m) => m.id === id ? updated : m))
    await saveCMSItem('team', updated)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Team Manager</h1>
          <p className="text-[#94A3B8] text-xs mt-1">Manage your agency&apos;s team members and their profiles.</p>
        </div>
        <button onClick={openNew} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all">
          <Plus size={16} /> Add Member
        </button>
      </div>

      <div className="relative max-w-sm">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search members..." className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50" />
      </div>

      {/* Team Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {filtered.map((member) => (
          <div key={member.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 text-center hover:border-[#273449]/80 transition-colors">
            {member.photo ? (
              <img src={member.photo} alt={member.name} className="w-16 h-16 rounded-2xl object-cover mx-auto mb-4 border border-blue-500/30 shadow-md" />
            ) : (
              <div className="w-16 h-16 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center mx-auto mb-4">
                <span className="font-display font-bold text-xl text-blue-400">
                  {member.name.split(' ').map((n) => n[0]).join('')}
                </span>
              </div>
            )}
            <h3 className="font-display text-sm font-bold text-white">{member.name}</h3>
            <p className="text-[10px] text-blue-400 font-semibold mb-2">{member.position}</p>
            <p className="text-[10px] text-[#94A3B8] mb-3 line-clamp-2">{member.bio}</p>

            <div className="flex items-center justify-center gap-1.5 mb-4">
              <button onClick={() => toggleStatus(member.id)} className={cn('px-2 py-0.5 rounded-full text-[8px] font-bold uppercase border', member.status === 'active' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' : 'bg-red-500/10 text-red-400 border-red-500/25')}>
                {member.status}
              </button>
              {member.featured && <span className="text-blue-500 text-[10px] font-bold">★ Featured</span>}
            </div>

            <div className="flex items-center justify-center gap-2 pt-3 border-t border-[#273449]/40">
              {member.linkedin && <a href={member.linkedin} target="_blank" rel="noreferrer" className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-blue-400 transition-colors"><Linkedin size={12} /></a>}
              {member.github && <a href={member.github} target="_blank" rel="noreferrer" className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-blue-400 transition-colors"><Github size={12} /></a>}
              {member.email && <a href={`mailto:${member.email}`} className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-blue-400 transition-colors"><Mail size={12} /></a>}
              <button onClick={() => openEdit(member)} className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-white transition-colors"><Pencil size={12} /></button>
              <button onClick={() => handleDelete(member.id)} className="p-1.5 rounded-lg bg-[#050816] border border-[#273449]/50 text-[#94A3B8] hover:text-red-400 transition-colors"><Trash2 size={12} /></button>
            </div>
          </div>
        ))}
        {filtered.length === 0 && <div className="col-span-full text-center py-12 text-[#94A3B8] text-sm">No team members found.</div>}
      </div>

      {/* Modal Editor */}
      {editing && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">{isNew ? 'Add Team Member' : 'Edit Member'}</h2>
              <button onClick={close} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white"><X size={18} /></button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Full Name *</label>
                  <input type="text" value={editing.name || ''} onChange={(e) => setEditing({ ...editing, name: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Position *</label>
                  <input type="text" value={editing.position || ''} onChange={(e) => setEditing({ ...editing, position: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
              </div>

              <ImagePicker
                label="Team Member Profile Photo (Optional)"
                value={editing.photo || null}
                onChange={(url) => setEditing({ ...editing, photo: url })}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Bio</label>
                <textarea rows={3} value={editing.bio || ''} onChange={(e) => setEditing({ ...editing, bio: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Email</label>
                  <input type="email" value={editing.email || ''} onChange={(e) => setEditing({ ...editing, email: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">LinkedIn URL</label>
                  <input type="text" value={editing.linkedin || ''} onChange={(e) => setEditing({ ...editing, linkedin: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">GitHub URL</label>
                  <input type="text" value={editing.github || ''} onChange={(e) => setEditing({ ...editing, github: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Featured?</label>
                  <div className="pt-1">
                    <button onClick={() => setEditing({ ...editing, featured: !editing.featured })} className={cn('w-12 h-6 rounded-full transition-colors relative', editing.featured ? 'bg-blue-600' : 'bg-[#273449]')}>
                      <span className={cn('absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform', editing.featured ? 'left-[26px]' : 'left-0.5')} />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449]">
              <button onClick={close} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">Cancel</button>
              <button onClick={handleSave} disabled={saving} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50">
                {saving ? <><Loader2 size={14} className="animate-spin" /> Saving...</> : <><Save size={14} /> {isNew ? 'Add Member' : 'Save Changes'}</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
