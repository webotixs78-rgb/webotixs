'use client'

import { useState, useEffect } from 'react'
import { Plus, Pencil, Trash2, Eye, EyeOff, GripVertical, Search, Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud, X, Save, Loader2 } from 'lucide-react'
import ImagePicker from '@/components/admin/ImagePicker'
import { mockServices } from '@/lib/data/mock'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'
import type { Service } from '@/lib/types'
import { cn } from '@/lib/utils'

const iconOptions = ['Globe', 'Smartphone', 'Palette', 'ShoppingCart', 'TrendingUp', 'Cloud']
const iconMap: Record<string, React.ComponentType<{ size?: number; className?: string }>> = { Globe, Smartphone, Palette, ShoppingCart, TrendingUp, Cloud }

const emptyService: Partial<Service> = {
  title: '', slug: '', icon: 'Globe', short_description: '', long_description: '',
  features: [], process: [], status: 'draft', featured: false, order: 0,
}

export default function AdminServicesPage() {
  const [services, setServices] = useState<Service[]>([...mockServices])
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<Partial<Service> | null>(null)
  const [isNew, setIsNew] = useState(false)
  const [saving, setSaving] = useState(false)
  const [featureInput, setFeatureInput] = useState('')

  useEffect(() => {
    getCMSData<Service[]>('services').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setServices(data)
      }
    })
  }, [])

  const filtered = services.filter((s) =>
    s.title.toLowerCase().includes(search.toLowerCase())
  )

  const openNew = () => { setEditing({ ...emptyService }); setIsNew(true) }
  const openEdit = (s: Service) => { setEditing({ ...s }); setIsNew(false) }
  const close = () => { setEditing(null); setIsNew(false); setFeatureInput('') }

  const handleSave = async () => {
    if (!editing?.title || !editing?.short_description || !editing?.long_description) return
    setSaving(true)

    const slug = editing.slug || editing.title.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '')
    const now = new Date().toISOString()

    if (isNew) {
      const newService: Service = {
        ...(editing as Service),
        id: crypto.randomUUID(),
        slug,
        features: editing.features || [],
        process: editing.process || [],
        gallery: [],
        seo: {},
        created_at: now,
        updated_at: now,
      }
      setServices((prev) => [newService, ...prev])
      await saveCMSItem('services', newService)
    } else {
      const updated = { ...editing, slug, updated_at: now } as Service
      setServices((prev) => prev.map((s) => s.id === editing.id ? updated : s))
      await saveCMSItem('services', updated)
    }
    setSaving(false)
    close()
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this service?')) return
    setServices((prev) => prev.filter((s) => s.id !== id))
    await deleteCMSItem('services', id)
  }

  const toggleStatus = async (id: string) => {
    const target = services.find((s) => s.id === id)
    if (!target) return
    const updated = { ...target, status: target.status === 'published' ? 'draft' : 'published' } as Service
    setServices((prev) => prev.map((s) => s.id === id ? updated : s))
    await saveCMSItem('services', updated)
  }

  const addFeature = () => {
    if (!featureInput.trim() || !editing) return
    setEditing({ ...editing, features: [...(editing.features || []), featureInput.trim()] })
    setFeatureInput('')
  }

  const removeFeature = (idx: number) => {
    if (!editing) return
    const f = [...(editing.features || [])]
    f.splice(idx, 1)
    setEditing({ ...editing, features: f })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Services Manager</h1>
          <p className="text-[#94A3B8] text-xs mt-1">Create, edit and manage your agency service offerings.</p>
        </div>
        <button onClick={openNew} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all">
          <Plus size={16} /> Add Service
        </button>
      </div>

      {/* Search */}
      <div className="relative max-w-sm">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search services..." className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50" />
      </div>

      {/* Table */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#273449] text-[#94A3B8] text-xs uppercase tracking-wider">
                <th className="text-left px-6 py-4 font-semibold">Service</th>
                <th className="text-left px-4 py-4 font-semibold hidden md:table-cell">Features</th>
                <th className="text-center px-4 py-4 font-semibold">Status</th>
                <th className="text-center px-4 py-4 font-semibold">Featured</th>
                <th className="text-right px-6 py-4 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((service) => {
                const Icon = iconMap[service.icon] ?? Globe
                return (
                  <tr key={service.id} className="border-b border-[#273449]/50 hover:bg-white/[0.02] transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-500">
                          <Icon size={16} />
                        </div>
                        <div>
                          <div className="font-semibold text-white">{service.title}</div>
                          <div className="text-[10px] text-[#94A3B8]">/{service.slug}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4 hidden md:table-cell">
                      <div className="flex flex-wrap gap-1">
                        {service.features.slice(0, 2).map((f) => (
                          <span key={f} className="px-2 py-0.5 bg-[#050816] border border-[#273449]/50 rounded text-[10px] text-[#94A3B8]">{f}</span>
                        ))}
                        {service.features.length > 2 && <span className="text-[10px] text-[#94A3B8]">+{service.features.length - 2}</span>}
                      </div>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <button onClick={() => toggleStatus(service.id)} className={cn('px-2.5 py-1 rounded-full text-[9px] font-bold uppercase border', service.status === 'published' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' : 'bg-amber-500/10 text-amber-400 border-amber-500/25')}>
                        {service.status}
                      </button>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <span className={cn('text-xs font-bold', service.featured ? 'text-blue-500' : 'text-[#94A3B8]/30')}>
                        {service.featured ? '★' : '—'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button onClick={() => openEdit(service)} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors" title="Edit">
                          <Pencil size={14} />
                        </button>
                        <button onClick={() => handleDelete(service.id)} className="p-2 rounded-lg hover:bg-red-500/10 text-[#94A3B8] hover:text-red-400 transition-colors" title="Delete">
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
              {filtered.length === 0 && (
                <tr><td colSpan={5} className="px-6 py-12 text-center text-[#94A3B8] text-sm">No services found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Editor */}
      {editing && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-2xl max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">{isNew ? 'Add New Service' : 'Edit Service'}</h2>
              <button onClick={close} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white"><X size={18} /></button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Title *</label>
                  <input type="text" value={editing.title || ''} onChange={(e) => setEditing({ ...editing, title: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="Web Design & Development" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Icon</label>
                  <select value={editing.icon || 'Globe'} onChange={(e) => setEditing({ ...editing, icon: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50">
                    {iconOptions.map((i) => <option key={i} value={i}>{i}</option>)}
                  </select>
                </div>
              </div>

              <ImagePicker
                label="Service Hero Banner / Thumbnail (Optional)"
                value={(editing as any).cover_image || null}
                onChange={(url) => setEditing({ ...editing, cover_image: url } as any)}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Short Description *</label>
                <textarea rows={2} value={editing.short_description || ''} onChange={(e) => setEditing({ ...editing, short_description: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" placeholder="Brief tagline for service cards..." />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Long Description *</label>
                <textarea rows={4} value={editing.long_description || ''} onChange={(e) => setEditing({ ...editing, long_description: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" placeholder="Detailed service overview..." />
              </div>

              {/* Features */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-[#94A3B8]">Features</label>
                <div className="flex gap-2">
                  <input type="text" value={featureInput} onChange={(e) => setFeatureInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addFeature())} className="flex-1 px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="Add a feature..." />
                  <button onClick={addFeature} className="px-4 py-2 bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded-xl hover:bg-blue-600/30 transition-colors">Add</button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {(editing.features || []).map((f, i) => (
                    <span key={i} className="flex items-center gap-1.5 px-3 py-1 bg-[#050816] border border-[#273449] rounded-lg text-xs text-[#94A3B8]">
                      {f}
                      <button onClick={() => removeFeature(i)} className="text-red-400 hover:text-red-300"><X size={10} /></button>
                    </span>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Status</label>
                  <select value={editing.status || 'draft'} onChange={(e) => setEditing({ ...editing, status: e.target.value as 'published' | 'draft' })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50">
                    <option value="draft">Draft</option>
                    <option value="published">Published</option>
                  </select>
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

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449]">
              <button onClick={close} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white hover:border-white/20 transition-colors">Cancel</button>
              <button onClick={handleSave} disabled={saving} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50">
                {saving ? <><Loader2 size={14} className="animate-spin" /> Saving...</> : <><Save size={14} /> {isNew ? 'Create Service' : 'Save Changes'}</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
