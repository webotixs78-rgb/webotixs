'use client'

import { useState, useEffect } from 'react'
import { Plus, Pencil, Trash2, Search, X, Save, Loader2, Calendar, Clock } from 'lucide-react'
import { RichTextEditor } from '@/components/admin/RichTextEditor'
import ImagePicker from '@/components/admin/ImagePicker'
import { mockBlogPosts, mockBlogCategories } from '@/lib/data/mock'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'
import type { BlogPost, BlogCategory } from '@/lib/types'
import { cn, formatDate } from '@/lib/utils'

const emptyPost: Partial<BlogPost> = {
  title: '', slug: '', excerpt: '', content: '', tags: [],
  status: 'draft', reading_time: 0, featured_image: null,
}

export default function AdminBlogsPage() {
  const [posts, setPosts] = useState<BlogPost[]>([...mockBlogPosts])
  const [categories] = useState<BlogCategory[]>([...mockBlogCategories])
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<Partial<BlogPost> | null>(null)
  const [isNew, setIsNew] = useState(false)
  const [saving, setSaving] = useState(false)
  const [tagInput, setTagInput] = useState('')

  useEffect(() => {
    getCMSData<BlogPost[]>('blogs').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setPosts(data)
      }
    })
  }, [])

  const filtered = posts.filter((p) => p.title.toLowerCase().includes(search.toLowerCase()))

  const openNew = () => { setEditing({ ...emptyPost }); setIsNew(true) }
  const openEdit = (p: BlogPost) => { setEditing({ ...p }); setIsNew(false) }
  const close = () => { setEditing(null); setIsNew(false); setTagInput('') }

  const handleSave = async () => {
    if (!editing?.title || !editing?.excerpt || !editing?.content) return
    setSaving(true)
    const now = new Date().toISOString()
    const slug = editing.slug || editing.title.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '')
    const wordCount = (editing.content || '').trim().split(/\s+/).length
    const reading_time = Math.ceil(wordCount / 200)

    if (isNew) {
      const newPost: BlogPost = {
        ...(editing as BlogPost),
        id: crypto.randomUUID(),
        slug,
        reading_time,
        featured_image: editing.featured_image || null,
        gallery: [],
        seo: {},
        related_posts: [],
        created_at: now,
        updated_at: now,
        published_at: editing.status === 'published' ? now : null,
      }
      setPosts((prev) => [newPost, ...prev])
      await saveCMSItem('blogs', newPost)
    } else {
      const updated = { ...editing, slug, reading_time, updated_at: now } as BlogPost
      setPosts((prev) => prev.map((p) => p.id === editing.id ? updated : p))
      await saveCMSItem('blogs', updated)
    }
    setSaving(false)
    close()
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this blog post?')) return
    setPosts((prev) => prev.filter((p) => p.id !== id))
    await deleteCMSItem('blogs', id)
  }

  const toggleStatus = async (id: string) => {
    const now = new Date().toISOString()
    const target = posts.find((p) => p.id === id)
    if (!target) return
    const updated = {
      ...target,
      status: target.status === 'published' ? 'draft' : 'published',
      published_at: target.status === 'draft' ? now : target.published_at,
    } as BlogPost
    setPosts((prev) => prev.map((p) => p.id === id ? updated : p))
    await saveCMSItem('blogs', updated)
  }

  const addTag = () => {
    if (!tagInput.trim() || !editing) return
    setEditing({ ...editing, tags: [...(editing.tags || []), tagInput.trim()] })
    setTagInput('')
  }
  const removeTag = (idx: number) => {
    if (!editing) return
    const t = [...(editing.tags || [])]
    t.splice(idx, 1)
    setEditing({ ...editing, tags: t })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Blog Manager</h1>
          <p className="text-[#94A3B8] text-xs mt-1">Write, edit, and publish blog posts and articles.</p>
        </div>
        <button onClick={openNew} className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all">
          <Plus size={16} /> New Post
        </button>
      </div>

      <div className="relative max-w-sm">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search posts..." className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50" />
      </div>

      {/* Posts Table */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#273449] text-[#94A3B8] text-xs uppercase tracking-wider">
                <th className="text-left px-6 py-4 font-semibold">Post</th>
                <th className="text-left px-4 py-4 font-semibold hidden md:table-cell">Category</th>
                <th className="text-center px-4 py-4 font-semibold hidden lg:table-cell">Reading</th>
                <th className="text-center px-4 py-4 font-semibold">Status</th>
                <th className="text-right px-6 py-4 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((post) => (
                <tr key={post.id} className="border-b border-[#273449]/50 hover:bg-white/[0.02] transition-colors">
                  <td className="px-6 py-4">
                    <div>
                      <div className="font-semibold text-white text-sm">{post.title}</div>
                      <div className="text-[10px] text-[#94A3B8] flex items-center gap-2 mt-0.5">
                        <Calendar size={10} />
                        {post.published_at ? formatDate(post.published_at) : 'Unpublished'}
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4 hidden md:table-cell">
                    <span className="px-2 py-0.5 bg-[#050816] border border-[#273449]/50 rounded text-[10px] text-[#94A3B8]">
                      {post.category?.name || '—'}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-center hidden lg:table-cell">
                    <span className="text-xs text-[#94A3B8] flex items-center justify-center gap-1">
                      <Clock size={11} /> {post.reading_time} min
                    </span>
                  </td>
                  <td className="px-4 py-4 text-center">
                    <button onClick={() => toggleStatus(post.id)} className={cn('px-2.5 py-1 rounded-full text-[9px] font-bold uppercase border', post.status === 'published' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' : 'bg-amber-500/10 text-amber-400 border-amber-500/25')}>
                      {post.status}
                    </button>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button onClick={() => openEdit(post)} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"><Pencil size={14} /></button>
                      <button onClick={() => handleDelete(post.id)} className="p-2 rounded-lg hover:bg-red-500/10 text-[#94A3B8] hover:text-red-400 transition-colors"><Trash2 size={14} /></button>
                    </div>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={5} className="px-6 py-12 text-center text-[#94A3B8] text-sm">No posts found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Editor */}
      {editing && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">{isNew ? 'Create New Post' : 'Edit Post'}</h2>
              <button onClick={close} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white"><X size={18} /></button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Title *</label>
                  <input type="text" value={editing.title || ''} onChange={(e) => setEditing({ ...editing, title: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Category</label>
                  <select value={editing.category?.id || ''} onChange={(e) => { const cat = categories.find((c) => c.id === e.target.value); setEditing({ ...editing, category: cat || undefined }) }} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50">
                    <option value="">Select category...</option>
                    {categories.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                </div>
              </div>

              <ImagePicker
                label="Blog Cover Image *"
                value={editing.featured_image || null}
                onChange={(url) => setEditing({ ...editing, featured_image: url })}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Excerpt *</label>
                <textarea rows={2} value={editing.excerpt || ''} onChange={(e) => setEditing({ ...editing, excerpt: e.target.value })} className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none" placeholder="Brief summary for listing cards..." />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8] flex items-center justify-between">
                  <span>Content * (Rich Text Editor)</span>
                  <span className="text-[10px] text-blue-400 font-normal">Tiptap Powered WYSIWYG</span>
                </label>
                <RichTextEditor
                  content={editing.content || ''}
                  onChange={(html) => setEditing({ ...editing, content: html })}
                />
              </div>

              {/* Tags */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-[#94A3B8]">Tags</label>
                <div className="flex gap-2">
                  <input type="text" value={tagInput} onChange={(e) => setTagInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())} className="flex-1 px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50" placeholder="nextjs, design..." />
                  <button onClick={addTag} className="px-4 py-2 bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded-xl">Add</button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(editing.tags || []).map((t, i) => (
                    <span key={i} className="flex items-center gap-1.5 px-3 py-1 bg-[#050816] border border-[#273449] rounded-lg text-xs text-[#94A3B8]">#{t}<button onClick={() => removeTag(i)} className="text-red-400"><X size={10} /></button></span>
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
                {saving ? <><Loader2 size={14} className="animate-spin" /> Saving...</> : <><Save size={14} /> {isNew ? 'Publish Post' : 'Save Changes'}</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
