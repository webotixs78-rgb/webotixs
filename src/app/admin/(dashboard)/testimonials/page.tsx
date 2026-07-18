'use client'

import { useState, useEffect } from 'react'
import { Plus, Edit, Trash2, Star, Quote, CheckCircle2, RefreshCw, MessageSquareQuote } from 'lucide-react'
import { mockTestimonials } from '@/lib/data/mock'
import { getCMSData, saveCMSList } from '@/lib/data/cms'
import type { Testimonial } from '@/lib/types'
import ImagePicker from '@/components/admin/ImagePicker'

export default function AdminTestimonialsPage() {
  const [items, setItems] = useState<Testimonial[]>(mockTestimonials)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingItem, setEditingItem] = useState<Testimonial | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  // Form states
  const [name, setName] = useState('')
  const [position, setPosition] = useState('')
  const [company, setCompany] = useState('')
  const [content, setContent] = useState('')
  const [rating, setRating] = useState(5)
  const [avatar, setAvatar] = useState<string | null>(null)

  useEffect(() => {
    getCMSData<Testimonial[]>('testimonials').then((parsed) => {
      if (Array.isArray(parsed) && parsed.length > 0) {
        setItems(parsed)
      }
    })
  }, [])

  const saveToStorage = async (newItems: Testimonial[]) => {
    setItems(newItems)
    await saveCMSList('testimonials', newItems)
  }

  const handleOpenAdd = () => {
    setEditingItem(null)
    setName('')
    setPosition('')
    setCompany('')
    setContent('')
    setRating(5)
    setAvatar(null)
    setIsModalOpen(true)
  }

  const handleOpenEdit = (item: Testimonial) => {
    setEditingItem(item)
    setName(item.name)
    setPosition(item.position)
    setCompany(item.company)
    setContent(item.content)
    setRating(item.rating || 5)
    setAvatar(item.avatar)
    setIsModalOpen(true)
  }

  const handleDelete = (id: string) => {
    if (!window.confirm('Delete this client testimonial from the live marquee sliders?')) return
    const filtered = items.filter((i) => i.id !== id)
    saveToStorage(filtered)
    setSuccessMsg('Testimonial removed successfully!')
    setTimeout(() => setSuccessMsg(null), 3000)
  }

  const handleSaveModal = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || !content.trim()) return

    if (editingItem) {
      const updated = items.map((i) =>
        i.id === editingItem.id
          ? { ...i, name, position, company, content, rating, avatar, updated_at: new Date().toISOString() }
          : i
      )
      saveToStorage(updated)
      setSuccessMsg('Testimonial updated and synced with homepage slider!')
    } else {
      const newItem: Testimonial = {
        id: Date.now().toString(),
        name,
        position,
        company,
        content,
        rating,
        avatar,
        featured: true,
        order: items.length + 1,
        created_at: new Date().toISOString(),
      }
      saveToStorage([newItem, ...items])
      setSuccessMsg('New testimonial created and added to homepage marquee!')
    }

    setIsModalOpen(false)
    setTimeout(() => setSuccessMsg(null), 3500)
  }

  const handleResetDefaults = () => {
    if (!window.confirm('Reset Testimonials CMS state to original production defaults?')) return
    localStorage.removeItem('webotixs_cms_testimonials')
    saveCMSList('testimonials', mockTestimonials).then(() => {
      setItems(mockTestimonials)
      setSuccessMsg('Reset to default testimonials!')
      setTimeout(() => setSuccessMsg(null), 3000)
    })
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#273449] pb-6">
        <div>
          <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2.5">
            <MessageSquareQuote size={24} className="text-blue-500" /> Client Testimonials & Reviews Manager
          </h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Add, edit, and organize quotes that appear in the dual-row infinite marquee sliders on your homepage.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleResetDefaults}
            className="px-4 py-2 bg-[#0D1224] border border-[#273449] text-[#94A3B8] hover:text-white text-xs font-semibold rounded-xl transition-colors flex items-center gap-1.5"
          >
            <RefreshCw size={13} /> Reset Defaults
          </button>
          <button
            onClick={handleOpenAdd}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-xl hover:shadow-glow-sm transition-all"
          >
            <Plus size={15} /> Add Testimonial
          </button>
        </div>
      </div>

      {successMsg && (
        <div className="px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold rounded-2xl flex items-center gap-2">
          <CheckCircle2 size={16} /> {successMsg}
        </div>
      )}

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {items.map((item) => (
          <div
            key={item.id}
            className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 flex flex-col justify-between hover:border-blue-500/40 transition-all space-y-4"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  {[...Array(item.rating || 5)].map((_, i) => (
                    <Star key={i} size={15} className="fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleOpenEdit(item)}
                    className="p-2 rounded-xl bg-[#050816] hover:bg-white/10 text-blue-400 text-xs font-semibold flex items-center gap-1 transition-colors"
                  >
                    <Edit size={13} /> Edit
                  </button>
                  <button
                    onClick={() => handleDelete(item.id)}
                    className="p-2 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-400 text-xs transition-colors"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>

              <p className="text-sm text-white font-medium leading-relaxed line-clamp-3 italic">
                &ldquo;{item.content}&rdquo;
              </p>
            </div>

            <div className="flex items-center gap-3 pt-3 border-t border-[#273449]">
              {item.avatar ? (
                <img src={item.avatar} alt={item.name} className="w-10 h-10 rounded-xl object-cover border border-[#273449]" />
              ) : (
                <div className="w-10 h-10 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 font-bold text-sm">
                  {item.name[0]}
                </div>
              )}
              <div className="min-w-0">
                <div className="text-xs font-bold text-white truncate">{item.name}</div>
                <div className="text-[11px] text-[#94A3B8] truncate">
                  {item.position}, {item.company}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Add / Edit Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#0D1224] border border-[#273449] rounded-3xl max-w-lg w-full max-h-[90vh] overflow-y-auto p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#273449] pb-4">
              <h3 className="font-display font-bold text-lg text-white flex items-center gap-2">
                <Quote size={18} className="text-blue-400" />
                {editingItem ? 'Edit Client Testimonial' : 'Add New Client Testimonial'}
              </h3>
            </div>

            <form onSubmit={handleSaveModal} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Client Name *</label>
                  <input
                    type="text"
                    required
                    placeholder="James Mitchell"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Rating (Stars)</label>
                  <select
                    value={rating}
                    onChange={(e) => setRating(Number(e.target.value))}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  >
                    <option value={5}>★★★★★ (5 Stars)</option>
                    <option value={4}>★★★★☆ (4 Stars)</option>
                    <option value={3}>★★★☆☆ (3 Stars)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Job Title / Position</label>
                  <input
                    type="text"
                    placeholder="Chief Technology Officer"
                    value={position}
                    onChange={(e) => setPosition(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Company Name</label>
                  <input
                    type="text"
                    placeholder="TechCorp Global"
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
              </div>

              {/* Avatar Picker */}
              <ImagePicker
                label="Client Avatar Photo (Optional)"
                value={avatar}
                onChange={(url) => setAvatar(url)}
              />

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Testimonial Quote *</label>
                <textarea
                  rows={4}
                  required
                  placeholder="Webotixs transformed our digital presence completely..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-[#273449]">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-5 py-2.5 bg-[#050816] hover:bg-white/5 text-[#94A3B8] text-xs font-semibold rounded-xl transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-bold text-xs rounded-xl hover:shadow-glow-sm transition-all"
                >
                  Save Testimonial
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
