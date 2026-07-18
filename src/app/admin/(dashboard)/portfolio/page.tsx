'use client'

import { useState, useEffect } from 'react'
import { Plus, Pencil, Trash2, Search, X, Save, Loader2, ExternalLink } from 'lucide-react'
import ImagePicker from '@/components/admin/ImagePicker'
import { mockPortfolio } from '@/lib/data/mock'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'
import type { PortfolioProject } from '@/lib/types'
import { cn } from '@/lib/utils'

const emptyProject: Partial<PortfolioProject> = {
  title: '', client: '', industry: '', description: '', challenge: '', solution: '',
  results: [], technologies: [], live_url: '', status: 'draft', featured: false, order: 0,
}

export default function AdminPortfolioPage() {
  const [projects, setProjects] = useState<PortfolioProject[]>([...mockPortfolio])
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<Partial<PortfolioProject> | null>(null)
  const [isNew, setIsNew] = useState(false)
  const [saving, setSaving] = useState(false)
  const [techInput, setTechInput] = useState('')
  const [resultInput, setResultInput] = useState('')

  useEffect(() => {
    getCMSData<PortfolioProject[]>('portfolio').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setProjects(data)
      }
    })
  }, [])

  const filtered = projects.filter((p) => p.title.toLowerCase().includes(search.toLowerCase()) || p.client.toLowerCase().includes(search.toLowerCase()))

  const openNew = () => { setEditing({ ...emptyProject }); setIsNew(true) }
  const openEdit = (p: PortfolioProject) => { setEditing({ ...p }); setIsNew(false) }
  const close = () => { setEditing(null); setIsNew(false); setTechInput(''); setResultInput('') }

  const handleSave = async () => {
    if (!editing?.title || !editing?.client || !editing?.description) return
    setSaving(true)
    const now = new Date().toISOString()
    if (isNew) {
      const newProj = { ...(editing as PortfolioProject), id: crypto.randomUUID(), thumbnail: editing.thumbnail || null, gallery: [], seo: {}, created_at: now, updated_at: now }
      setProjects((prev) => [newProj, ...prev])
      await saveCMSItem('portfolio', newProj)
    } else {
      const updated = { ...editing, updated_at: now } as PortfolioProject
      setProjects((prev) => prev.map((p) => p.id === editing.id ? updated : p))
      await saveCMSItem('portfolio', updated)
    }
    setSaving(false)
    close()
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this project?')) return
    setProjects((prev) => prev.filter((p) => p.id !== id))
    await deleteCMSItem('portfolio', id)
  }

  const toggleStatus = async (id: string) => {
    const target = projects.find((p) => p.id === id)
    if (!target) return
    const updated = { ...target, status: target.status === 'published' ? 'draft' : 'published' } as PortfolioProject
    setProjects((prev) => prev.map((p) => p.id === id ? updated : p))
    await saveCMSItem('portfolio', updated)
  }

  const addTag = (field: 'technologies' | 'results', input: string, setInput: (v: string) => void) => {
    if (!input.trim() || !editing) return
    setEditing({ ...editing, [field]: [...((editing[field] as string[]) || []), input.trim()] })
    setInput('')
  }

  const removeTag = (field: 'technologies' | 'results', idx: number) => {
    if (!editing) return
    const arr = [...((editing[field] as string[]) || [])]
    arr.splice(idx, 1)
    setEditing({ ...editing, [field]: arr })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Portfolio Manager</h1>
          <p className="text-[#94A3B8] text-xs mt-1">Manage your case studies and project showcase.</p>
        </div>
        <button onClick={openNew} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all">
          <Plus size={16} /> Add Project
        </button>
      </div>

      <div className="relative max-w-sm">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search projects..." className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50" />
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filtered.map((project) => (
          <div key={project.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden hover:border-[#273449]/80 transition-colors">
            <div className="h-36 bg-gradient-to-br from-[#0D1224] to-[#050816] flex items-center justify-center border-b border-[#273449]/50">
              <div className="w-14 h-14 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center">
                <span className="text-xl font-bold text-blue-400">{project.title[0]}</span>
              </div>
            </div>
            <div className="p-5 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h3 className="font-display text-sm font-bold text-white">{project.title}</h3>
                  <p className="text-[10px] text-[#94A3B8]">{project.client} · {project.industry}</p>
                </div>
                <button onClick={() => toggleStatus(project.id)} className={cn('px-2 py-0.5 rounded-full text-[8px] font-bold uppercase border flex-shrink-0', project.status === 'published' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' : 'bg-amber-500/10 text-amber-400 border-amber-500/25')}>
                  {project.status}
                </button>
              </div>

              <div className="flex flex-wrap gap-1">
                {project.technologies.slice(0, 3).map((t) => (
                  <span key={t} className="px-2 py-0.5 bg-[#050816] border border-[#273449]/50 rounded text-[9px] text-[#94A3B8]">{t}</span>
                ))}
              </div>

              <div className="flex items-center justify-end gap-1.5 pt-2 border-t border-[#273449]/40">
                <button onClick={() => openEdit(project)} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"><Pencil size={13} /></button>
                <button onClick={() => handleDelete(project.id)} className="p-2 rounded-lg hover:bg-red-500/10 text-[#94A3B8] hover:text-red-400 transition-colors"><Trash2 size={13} /></button>
              </div>
            </div>
          </div>
        ))}
        {filtered.length === 0 && <div className="col-span-full text-center py-12 text-[#94A3B8] text-sm">No projects found.</div>}
      </div>

      {/* Modal Editor */}
      {editing && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-2xl max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">{isNew ? 'Add Project' : 'Edit Project'}</h2>
              <button onClick={close} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white"><X size={18} /></button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Project Title *</label>
                  <input type="text" value={editing.title || ''} onChange={(e) => setEditing({ ...editing, title: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Client Name *</label>
                  <input type="text" value={editing.client || ''} onChange={(e) => setEditing({ ...editing, client: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Industry</label>
                  <input type="text" value={editing.industry || ''} onChange={(e) => setEditing({ ...editing, industry: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Live URL</label>
                  <input type="text" value={editing.live_url || ''} onChange={(e) => setEditing({ ...editing, live_url: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="https://" />
                </div>
              </div>

              <ImagePicker
                label="Case Study Thumbnail / Hero Cover *"
                value={editing.thumbnail || null}
                onChange={(url) => setEditing({ ...editing, thumbnail: url })}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Description *</label>
                <textarea rows={3} value={editing.description || ''} onChange={(e) => setEditing({ ...editing, description: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Challenge</label>
                  <textarea rows={3} value={editing.challenge || ''} onChange={(e) => setEditing({ ...editing, challenge: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Solution</label>
                  <textarea rows={3} value={editing.solution || ''} onChange={(e) => setEditing({ ...editing, solution: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" />
                </div>
              </div>

              {/* Technologies */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-[#94A3B8]">Technologies</label>
                <div className="flex gap-2">
                  <input type="text" value={techInput} onChange={(e) => setTechInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag('technologies', techInput, setTechInput))} className="flex-1 px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="Next.js, React..." />
                  <button onClick={() => addTag('technologies', techInput, setTechInput)} className="px-4 py-2 bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded-xl">Add</button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(editing.technologies || []).map((t, i) => (
                    <span key={i} className="flex items-center gap-1.5 px-3 py-1 bg-[#050816] border border-[#273449] rounded-lg text-xs text-[#94A3B8]">{t}<button onClick={() => removeTag('technologies', i)} className="text-red-400"><X size={10} /></button></span>
                  ))}
                </div>
              </div>

              {/* Results */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-[#94A3B8]">Key Results</label>
                <div className="flex gap-2">
                  <input type="text" value={resultInput} onChange={(e) => setResultInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag('results', resultInput, setResultInput))} className="flex-1 px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="99.9% uptime..." />
                  <button onClick={() => addTag('results', resultInput, setResultInput)} className="px-4 py-2 bg-emerald-600/20 border border-emerald-500/30 text-emerald-400 text-xs font-bold rounded-xl">Add</button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(editing.results || []).map((r, i) => (
                    <span key={i} className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-xs text-emerald-400">{r}<button onClick={() => removeTag('results', i)} className="text-red-400"><X size={10} /></button></span>
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

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449]">
              <button onClick={close} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">Cancel</button>
              <button onClick={handleSave} disabled={saving} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50">
                {saving ? <><Loader2 size={14} className="animate-spin" /> Saving...</> : <><Save size={14} /> {isNew ? 'Create Project' : 'Save Changes'}</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
