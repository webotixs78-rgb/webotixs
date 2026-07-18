'use client'

import { useState, useEffect } from 'react'
import { Upload, Image as ImageIcon, Check, X, FolderOpen, Sparkles, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { createClient } from '@/lib/supabase/client'

interface ImagePickerProps {
  value: string | null
  onChange: (url: string | null) => void
  label?: string
}

const defaultMediaPresets = [
  { id: 'm1', title: 'Modern Dark Tech UI', url: 'https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=1200&auto=format&fit=crop' },
  { id: 'm2', title: 'Corporate Meeting Room', url: 'https://images.unsplash.com/photo-1497366216548-37526070297c?q=80&w=1200&auto=format&fit=crop' },
  { id: 'm3', title: 'App Development Code', url: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=1200&auto=format&fit=crop' },
  { id: 'm4', title: 'Futuristic AI Brain Canvas', url: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1200&auto=format&fit=crop' },
  { id: 'm5', title: 'Executive CEO Portrait', url: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=800&auto=format&fit=crop' },
  { id: 'm6', title: 'Senior UX Designer Portrait', url: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=800&auto=format&fit=crop' },
]

export default function ImagePicker({ value, onChange, label = 'Cover / Thumbnail Image' }: ImagePickerProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<'upload' | 'library' | 'url'>('upload')
  const [customUrl, setCustomUrl] = useState('')
  const [uploading, setUploading] = useState(false)
  const [mediaItems, setMediaItems] = useState<{ id: string; title: string; url: string }[]>(defaultMediaPresets)
  const supabase = createClient()

  useEffect(() => {
    try {
      const stored = localStorage.getItem('webotixs_cms_media')
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed) && parsed.length > 0) {
          const formatted = parsed.map((item: any) => ({
            id: item.id || crypto.randomUUID(),
            title: item.name || item.title || 'Uploaded Asset',
            url: item.url,
          }))
          setMediaItems([...formatted, ...defaultMediaPresets])
        }
      }
    } catch {}
  }, [isOpen])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    const fileName = `${Date.now()}-${file.name.replace(/\s+/g, '-')}`
    const filePath = `uploads/${fileName}`

    let finalUrl: string | null = null

    // 1. Attempt upload to Supabase storage bucket 'media'
    try {
      const { error } = await supabase.storage.from('media').upload(filePath, file)
      if (!error) {
        const { data } = supabase.storage.from('media').getPublicUrl(filePath)
        if (data?.publicUrl) {
          finalUrl = data.publicUrl
        }
      }
    } catch (err) {
      // Fallback if Supabase storage is not configured or offline
    }

    // 2. If Supabase upload didn't return a publicUrl, fallback to base64 Data URL
    if (!finalUrl) {
      await new Promise<void>((resolve) => {
        const reader = new FileReader()
        reader.onload = () => {
          finalUrl = reader.result as string
          resolve()
        }
        reader.readAsDataURL(file)
      })
    }

    if (finalUrl) {
      // Add to local storage media list
      try {
        const stored = localStorage.getItem('webotixs_cms_media')
        const currentList = stored ? JSON.parse(stored) : []
        const newItem = {
          id: crypto.randomUUID(),
          name: file.name,
          url: finalUrl,
          size: file.size,
          type: 'image',
          folder: 'Backgrounds',
          created_at: new Date().toISOString(),
        }
        if (Array.isArray(currentList)) {
          localStorage.setItem('webotixs_cms_media', JSON.stringify([newItem, ...currentList]))
        }
      } catch {}

      onChange(finalUrl)
    }

    setUploading(false)
    setIsOpen(false)
  }

  const handleSelectFromLibrary = (url: string) => {
    onChange(url)
    setIsOpen(false)
  }

  const handleApplyCustomUrl = () => {
    if (customUrl.trim()) {
      onChange(customUrl.trim())
      setCustomUrl('')
      setIsOpen(false)
    }
  }

  return (
    <div className="space-y-2">
      <label className="text-xs font-semibold text-[#94A3B8] flex items-center justify-between">
        <span>{label}</span>
        {value && (
          <button
            type="button"
            onClick={() => onChange(null)}
            className="text-[10px] text-red-400 hover:underline"
          >
            Remove Image
          </button>
        )}
      </label>

      {/* Preview Box / Trigger */}
      <div
        onClick={() => setIsOpen(true)}
        className={cn(
          'relative border-2 border-dashed rounded-2xl p-4 cursor-pointer transition-all flex items-center justify-center min-h-[120px] overflow-hidden group',
          value
            ? 'border-blue-500/50 bg-[#0A0E1F]'
            : 'border-[#273449] hover:border-blue-500/40 bg-[#050816]'
        )}
      >
        {value ? (
          <>
            <img src={value} alt="Preview" className="absolute inset-0 w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform" />
            <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
              <span className="px-3 py-1.5 bg-blue-600 text-white text-xs font-bold rounded-xl shadow-lg flex items-center gap-1.5">
                <Upload size={13} /> Change Image
              </span>
            </div>
          </>
        ) : (
          <div className="text-center space-y-2 py-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mx-auto group-hover:scale-110 transition-transform">
              <Upload size={18} />
            </div>
            <div className="text-xs font-bold text-white">Click to Upload or Select Image</div>
            <p className="text-[10px] text-[#94A3B8]">Supports local device upload, Media Library picker, or URL</p>
          </div>
        )}
      </div>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#0D1224] border border-[#273449] rounded-3xl max-w-2xl w-full max-h-[85vh] flex flex-col overflow-hidden shadow-2xl">
            {/* Modal Header */}
            <div className="p-5 border-b border-[#273449] flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center text-white">
                  <ImageIcon size={18} />
                </div>
                <div>
                  <h3 className="font-display font-bold text-base text-white">Select or Upload Media</h3>
                  <p className="text-[11px] text-[#94A3B8]">Pick from your assets or upload from device</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="w-8 h-8 rounded-xl bg-white/5 hover:bg-white/10 flex items-center justify-center text-[#94A3B8] hover:text-white transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Tabs Bar */}
            <div className="flex border-b border-[#273449] px-5 bg-[#050816]">
              {[
                { id: 'upload', label: 'Upload from Device', icon: Upload },
                { id: 'library', label: 'Media Library', icon: FolderOpen },
                { id: 'url', label: 'Custom URL', icon: Sparkles },
              ].map((tab) => {
                const Icon = tab.icon
                const isActive = activeTab === tab.id
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id as any)}
                    className={cn(
                      'flex items-center gap-2 px-4 py-3 text-xs font-bold border-b-2 transition-all',
                      isActive
                        ? 'border-blue-500 text-white bg-blue-600/10'
                        : 'border-transparent text-[#94A3B8] hover:text-white'
                    )}
                  >
                    <Icon size={14} className={isActive ? 'text-blue-400' : ''} />
                    {tab.label}
                  </button>
                )
              })}
            </div>

            {/* Modal Body */}
            <div className="p-6 flex-1 overflow-y-auto">
              {activeTab === 'upload' && (
                <div className="space-y-4">
                  <label className="border-2 border-dashed border-[#273449] hover:border-blue-500/50 rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer bg-[#050816] group transition-colors">
                    <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                    {uploading ? (
                      <div className="flex flex-col items-center gap-3">
                        <Loader2 size={28} className="animate-spin text-blue-500" />
                        <span className="text-xs font-bold text-white">Processing & Converting Image...</span>
                      </div>
                    ) : (
                      <div className="text-center space-y-3">
                        <div className="w-14 h-14 rounded-2xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-400 mx-auto group-hover:scale-110 transition-transform">
                          <Upload size={24} />
                        </div>
                        <div>
                          <div className="text-sm font-bold text-white">Browse Device or Drag & Drop File</div>
                          <p className="text-xs text-[#94A3B8] mt-1">PNG, JPG, WEBP, or SVG up to 10MB</p>
                        </div>
                      </div>
                    )}
                  </label>
                </div>
              )}

              {activeTab === 'library' && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {mediaItems.map((m, idx) => (
                    <div
                      key={`${m.id}-${idx}`}
                      onClick={() => handleSelectFromLibrary(m.url)}
                      className="group relative border border-[#273449] hover:border-blue-500 rounded-2xl overflow-hidden cursor-pointer aspect-video bg-[#050816] transition-all"
                    >
                      <img src={m.url} alt={m.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-90 flex items-end p-2.5">
                        <span className="text-[11px] font-bold text-white truncate w-full">{m.title}</span>
                      </div>
                      <div className="absolute top-2 right-2 w-6 h-6 rounded-lg bg-blue-600 text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity shadow-lg">
                        <Check size={12} />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'url' && (
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#94A3B8]">Paste Direct Image URL</label>
                    <input
                      type="url"
                      placeholder="https://images.unsplash.com/..."
                      value={customUrl}
                      onChange={(e) => setCustomUrl(e.target.value)}
                      className="w-full px-4 py-3 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 font-mono"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={handleApplyCustomUrl}
                    disabled={!customUrl.trim()}
                    className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-bold text-xs rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50"
                  >
                    Apply Image URL
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
