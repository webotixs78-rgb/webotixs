'use client'

import { useState, useEffect } from 'react'
import { Plus, Pencil, Trash2, Search, X, Save, Loader2 } from 'lucide-react'
import ImagePicker from '@/components/admin/ImagePicker'
import { cn } from '@/lib/utils'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'

interface Industry {
  id: string; title: string; slug: string; description: string; benefits: string[]
  status: 'published' | 'draft'; order: number; created_at: string; image?: string | null
}

const mockIndustries: Industry[] = [
  { id: '1', title: 'Fintech & Banking', slug: 'fintech-banking', description: 'Secure, high-frequency transaction applications.', benefits: ['End-to-end encryption', '99.99% uptime', 'Compliance built-in'], status: 'published', order: 1, created_at: new Date().toISOString() },
  { id: '2', title: 'E-Commerce & Retail', slug: 'ecommerce-retail', description: 'High-speed headless commerce stores.', benefits: ['Sub-second loads', 'AI recommender', 'Multi-currency'], status: 'published', order: 2, created_at: new Date().toISOString() },
  { id: '3', title: 'Healthcare & Telemedicine', slug: 'healthcare', description: 'HIPAA-compliant patient portals.', benefits: ['HIPAA compliant', 'Encrypted databases', 'Video consults'], status: 'published', order: 3, created_at: new Date().toISOString() },
  { id: '4', title: 'EdTech & E-Learning', slug: 'edtech', description: 'Custom learning management systems.', benefits: ['Global CDN', 'Adaptive pipelines', 'Automated quizzing'], status: 'draft', order: 4, created_at: new Date().toISOString() },
]

const emptyIndustry: Partial<Industry> = { title: '', slug: '', description: '', benefits: [], status: 'draft', order: 0 }

export default function AdminIndustriesPage() {
  const [industries, setIndustries] = useState<Industry[]>([...mockIndustries])
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<Partial<Industry> | null>(null)
  const [isNew, setIsNew] = useState(false)
  const [saving, setSaving] = useState(false)
  const [benefitInput, setBenefitInput] = useState('')

  useEffect(() => {
    getCMSData<Industry[]>('industries').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setIndustries(data)
      }
    }).catch(() => {})
  }, [])

  const filtered = industries.filter((i) => (i.title || '').toLowerCase().includes(search.toLowerCase()))

  const openNew = () => { setEditing({ ...emptyIndustry }); setIsNew(true) }
  const openEdit = (i: Industry) => { setEditing({ ...i }); setIsNew(false) }
  const close = () => { setEditing(null); setIsNew(false); setBenefitInput('') }

  const handleSave = async () => {
    if (!editing?.title || !editing?.description) return
    setSaving(true)
    const now = new Date().toISOString()
    const slug = editing.slug || editing.title.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '')
    if (isNew) {
      const newInd = { ...(editing as Industry), id: crypto.randomUUID(), slug, created_at: now }
      setIndustries((prev) => [...prev, newInd])
      await saveCMSItem('industries', newInd)
    } else {
      const updatedInd = { ...editing, slug } as Industry
      setIndustries((prev) => prev.map((i) => i.id === editing.id ? updatedInd : i))
      await saveCMSItem('industries', updatedInd)
    }
    setSaving(false)
    close()
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this industry?')) return
    setIndustries((prev) => prev.filter((i) => i.id !== id))
    await deleteCMSItem('industries', id)
  }

  const toggleStatus = async (id: string) => {
    const target = industries.find((i) => i.id === id)
    if (!target) return
    const updatedInd = { ...target, status: target.status === 'published' ? 'draft' : 'published' } as Industry
    setIndustries((prev) => prev.map((i) => i.id === id ? updatedInd : i))
    await saveCMSItem('industries', updatedInd)
  }

  const addBenefit = () => {
    if (!benefitInput.trim() || !editing) return
    setEditing({ ...editing, benefits: [...(editing.benefits || []), benefitInput.trim()] })
    setBenefitInput('')
  }
  const removeBenefit = (idx: number) => {
    if (!editing) return
    const b = [...(editing.benefits || [])]; b.splice(idx, 1)
    setEditing({ ...editing, benefits: b })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Industries Manager</h1>
          <p className="text-[#94A3B8] text-xs mt-1">Manage industry verticals and their solution offerings.</p>
        </div>
        <button onClick={openNew} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all">
          <Plus size={16} /> Add Industry
        </button>
      </div>

      <div className="relative max-w-sm">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search industries..." className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50" />
      </div>

      <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#273449] text-[#94A3B8] text-xs uppercase tracking-wider">
                <th className="text-left px-6 py-4 font-semibold">Industry</th>
                <th className="text-left px-4 py-4 font-semibold hidden md:table-cell">Benefits</th>
                <th className="text-center px-4 py-4 font-semibold">Status</th>
                <th className="text-right px-6 py-4 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ind) => (
                <tr key={ind.id} className="border-b border-[#273449]/50 hover:bg-white/[0.02] transition-colors">
                  <td className="px-6 py-4">
                    <div className="font-semibold text-white">{ind.title}</div>
                    <div className="text-[10px] text-[#94A3B8]">/{ind.slug}</div>
                  </td>
                  <td className="px-4 py-4 hidden md:table-cell">
                    <div className="flex flex-wrap gap-1">
                      {(ind.benefits || []).slice(0, 2).map((b) => (
                        <span key={b} className="px-2 py-0.5 bg-[#050816] border border-[#273449]/50 rounded text-[10px] text-[#94A3B8]">{b}</span>
                      ))}
                      {(ind.benefits || []).length > 2 && <span className="text-[10px] text-[#94A3B8]">+{(ind.benefits || []).length - 2}</span>}
                    </div>
                  </td>
                  <td className="px-4 py-4 text-center">
                    <button onClick={() => toggleStatus(ind.id)} className={cn('px-2.5 py-1 rounded-full text-[9px] font-bold uppercase border', ind.status === 'published' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' : 'bg-amber-500/10 text-amber-400 border-amber-500/25')}>
                      {ind.status}
                    </button>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button onClick={() => openEdit(ind)} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"><Pencil size={14} /></button>
                      <button onClick={() => handleDelete(ind.id)} className="p-2 rounded-lg hover:bg-red-500/10 text-[#94A3B8] hover:text-red-400 transition-colors"><Trash2 size={14} /></button>
                    </div>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && <tr><td colSpan={4} className="px-6 py-12 text-center text-[#94A3B8] text-sm">No industries found.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Editor */}
      {editing && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">{isNew ? 'Add Industry' : 'Edit Industry'}</h2>
              <button onClick={close} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white"><X size={18} /></button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Title *</label>
                <input type="text" value={editing.title || ''} onChange={(e) => setEditing({ ...editing, title: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
              </div>

              <ImagePicker
                label="Industry Cover / Hero Banner (Optional)"
                value={editing.image || null}
                onChange={(url) => setEditing({ ...editing, image: url })}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Description *</label>
                <textarea rows={4} value={editing.description || ''} onChange={(e) => setEditing({ ...editing, description: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold text-[#94A3B8]">Benefits / Key Solutions</label>
                <div className="flex gap-2">
                  <input type="text" value={benefitInput} onChange={(e) => setBenefitInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addBenefit())} className="flex-1 px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="Add benefit..." />
                  <button onClick={addBenefit} className="px-4 py-2 bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded-xl">Add</button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(editing.benefits || []).map((b, i) => (
                    <span key={i} className="flex items-center gap-1.5 px-3 py-1 bg-[#050816] border border-[#273449] rounded-lg text-xs text-[#94A3B8]">{b}<button onClick={() => removeBenefit(i)} className="text-red-400"><X size={10} /></button></span>
                  ))}
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Status</label>
                <select value={editing.status || 'draft'} onChange={(e) => setEditing({ ...editing, status: e.target.value as 'published' | 'draft' })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50">
                  <option value="draft">Draft</option>
                  <option value="published">Published</option>
                </select>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449]">
              <button onClick={close} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">Cancel</button>
              <button onClick={handleSave} disabled={saving} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50">
                {saving ? <><Loader2 size={14} className="animate-spin" /> Saving...</> : <><Save size={14} /> {isNew ? 'Create Industry' : 'Save Changes'}</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
