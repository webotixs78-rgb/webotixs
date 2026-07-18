'use client'

import { useState, useRef } from 'react'
import {
  Upload,
  FolderPlus,
  Search,
  Trash2,
  Copy,
  Check,
  Image as ImageIcon,
  FileText,
  Video,
  Folder,
  X,
  Loader2,
  ExternalLink,
  Filter,
  Grid,
  List as ListIcon,
} from 'lucide-react'
import { cn, formatFileSize, formatDateShort } from '@/lib/utils'
import { createClient } from '@/lib/supabase/client'

interface MediaItem {
  id: string
  name: string
  url: string
  size: number
  type: 'image' | 'video' | 'document'
  folder: string
  created_at: string
}

const mockMediaItems: MediaItem[] = [
  {
    id: '1',
    name: 'hero-mesh-background.png',
    url: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=1200&auto=format&fit=crop',
    size: 1450200,
    type: 'image',
    folder: 'Backgrounds',
    created_at: new Date().toISOString(),
  },
  {
    id: '2',
    name: 'fintech-dashboard-mockup.png',
    url: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop',
    size: 2840100,
    type: 'image',
    folder: 'Portfolio',
    created_at: new Date().toISOString(),
  },
  {
    id: '3',
    name: 'team-ceo-portrait.jpg',
    url: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&auto=format&fit=crop',
    size: 920400,
    type: 'image',
    folder: 'Team',
    created_at: new Date().toISOString(),
  },
  {
    id: '4',
    name: 'agency-brand-guidelines.pdf',
    url: '#',
    size: 4510000,
    type: 'document',
    folder: 'Documents',
    created_at: new Date().toISOString(),
  },
  {
    id: '5',
    name: 'product-demo-reel.mp4',
    url: '#',
    size: 18450000,
    type: 'video',
    folder: 'Videos',
    created_at: new Date().toISOString(),
  },
]

const initialFolders = ['All Media', 'Backgrounds', 'Portfolio', 'Team', 'Documents', 'Videos']

export default function AdminMediaPage() {
  const [items, setItems] = useState<MediaItem[]>([...mockMediaItems])
  const [folders, setFolders] = useState<string[]>([...initialFolders])
  const [activeFolder, setActiveFolder] = useState<string>('All Media')
  const [search, setSearch] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [isUploading, setIsUploading] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [showFolderModal, setShowFolderModal] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const supabase = createClient()

  const filteredItems = items.filter((item) => {
    const matchesFolder = activeFolder === 'All Media' || item.folder === activeFolder
    const matchesSearch = item.name.toLowerCase().includes(search.toLowerCase())
    return matchesFolder && matchesSearch
  })

  const handleCopy = (id: string, url: string) => {
    navigator.clipboard.writeText(url)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const handleDelete = (id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id))
  }

  const handleCreateFolder = () => {
    if (!newFolderName.trim() || folders.includes(newFolderName.trim())) return
    setFolders((prev) => [...prev, newFolderName.trim()])
    setActiveFolder(newFolderName.trim())
    setNewFolderName('')
    setShowFolderModal(false)
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const fileExt = file.name.split('.').pop()
      const fileName = `${Date.now()}-${file.name.replace(/\s+/g, '-')}`
      const filePath = `${activeFolder === 'All Media' ? 'uploads' : activeFolder}/${fileName}`

      // Attempt Supabase storage upload if connected
      let publicUrl = URL.createObjectURL(file)
      try {
        const { error } = await supabase.storage.from('media').upload(filePath, file)
        if (!error) {
          const { data } = supabase.storage.from('media').getPublicUrl(filePath)
          if (data?.publicUrl) publicUrl = data.publicUrl
        }
      } catch (err) {
        // Fallback if bucket not available in dev
      }

      let type: 'image' | 'video' | 'document' = 'document'
      if (file.type.startsWith('image/')) type = 'image'
      else if (file.type.startsWith('video/')) type = 'video'

      const newItem: MediaItem = {
        id: crypto.randomUUID(),
        name: file.name,
        url: publicUrl,
        size: file.size,
        type,
        folder: activeFolder === 'All Media' ? 'Backgrounds' : activeFolder,
        created_at: new Date().toISOString(),
      }

      setItems((prev) => [newItem, ...prev])
    }

    setIsUploading(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white">Media Library</h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Manage images, videos, and documents with Supabase Storage buckets.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setShowFolderModal(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-[#0D1224] border border-[#273449] text-white text-sm font-semibold rounded-xl hover:bg-white/5 transition-colors"
          >
            <FolderPlus size={16} className="text-blue-400" /> New Folder
          </button>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileUpload}
            className="hidden"
            accept="image/*,video/*,.pdf,.doc,.docx"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50"
          >
            {isUploading ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Uploading...
              </>
            ) : (
              <>
                <Upload size={16} /> Upload Media
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Folders Sidebar */}
        <div className="lg:col-span-1 bg-[#0D1224] border border-[#273449] rounded-2xl p-4 space-y-2 h-fit">
          <div className="text-xs font-bold text-[#94A3B8] uppercase tracking-wider px-3 py-2">Folders</div>
          <div className="space-y-1">
            {folders.map((folder) => {
              const count =
                folder === 'All Media' ? items.length : items.filter((i) => i.folder === folder).length
              return (
                <button
                  key={folder}
                  onClick={() => setActiveFolder(folder)}
                  className={cn(
                    'flex items-center justify-between w-full px-3 py-2.5 rounded-xl text-xs font-medium transition-colors text-left',
                    activeFolder === folder
                      ? 'bg-blue-600 text-white shadow-glow-sm'
                      : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
                  )}
                >
                  <span className="flex items-center gap-2.5 truncate">
                    <Folder size={15} className={activeFolder === folder ? 'text-white' : 'text-blue-500'} />
                    {folder}
                  </span>
                  <span
                    className={cn(
                      'px-2 py-0.5 rounded-full text-[10px] font-bold',
                      activeFolder === folder ? 'bg-white/20 text-white' : 'bg-[#050816] text-[#94A3B8]'
                    )}
                  >
                    {count}
                  </span>
                </button>
              )
            })}
          </div>
        </div>

        {/* Media Grid / Table */}
        <div className="lg:col-span-4 space-y-4">
          {/* Controls Bar */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-[#0D1224] border border-[#273449] rounded-2xl p-4">
            <div className="relative w-full sm:w-72">
              <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search files..."
                className="w-full pl-10 pr-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50"
              />
            </div>

            <div className="flex items-center gap-2 self-end sm:self-auto">
              <button
                onClick={() => setViewMode('grid')}
                className={cn(
                  'p-2 rounded-xl border transition-colors',
                  viewMode === 'grid'
                    ? 'bg-blue-600/20 border-blue-500/40 text-blue-400'
                    : 'bg-[#050816] border-[#273449] text-[#94A3B8] hover:text-white'
                )}
                title="Grid View"
              >
                <Grid size={16} />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={cn(
                  'p-2 rounded-xl border transition-colors',
                  viewMode === 'list'
                    ? 'bg-blue-600/20 border-blue-500/40 text-blue-400'
                    : 'bg-[#050816] border-[#273449] text-[#94A3B8] hover:text-white'
                )}
                title="List View"
              >
                <ListIcon size={16} />
              </button>
            </div>
          </div>

          {/* Grid View */}
          {viewMode === 'grid' ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              {filteredItems.map((item) => (
                <div
                  key={item.id}
                  className="group bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden hover:border-blue-500/50 transition-all flex flex-col"
                >
                  {/* Preview */}
                  <div className="h-36 bg-[#050816] relative flex items-center justify-center overflow-hidden border-b border-[#273449]">
                    {item.type === 'image' && item.url.startsWith('http') ? (
                      <img src={item.url} alt={item.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                    ) : (
                      <div className="flex flex-col items-center justify-center gap-2 text-[#94A3B8]">
                        {item.type === 'image' && <ImageIcon size={32} className="text-blue-500" />}
                        {item.type === 'video' && <Video size={32} className="text-cyan-500" />}
                        {item.type === 'document' && <FileText size={32} className="text-violet-500" />}
                        <span className="text-[10px] uppercase font-bold text-[#94A3B8]/60">{item.type}</span>
                      </div>
                    )}

                    {/* Actions overlay */}
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                      <button
                        onClick={() => handleCopy(item.id, item.url)}
                        className="p-2 rounded-xl bg-white/10 hover:bg-white/20 text-white backdrop-blur-md transition-colors"
                        title="Copy URL"
                      >
                        {copiedId === item.id ? <Check size={16} className="text-emerald-400" /> : <Copy size={16} />}
                      </button>
                      <button
                        onClick={() => handleDelete(item.id)}
                        className="p-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 text-red-400 backdrop-blur-md transition-colors"
                        title="Delete File"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>

                  {/* Info */}
                  <div className="p-3.5 flex-1 flex flex-col justify-between">
                    <div>
                      <div className="text-xs font-semibold text-white truncate" title={item.name}>
                        {item.name}
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-[#94A3B8] mt-1">
                        <span>{formatFileSize(item.size)}</span>
                        <span>{formatDateShort(item.created_at)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* List View */
            <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[#273449] text-[#94A3B8] text-xs uppercase tracking-wider">
                      <th className="text-left px-6 py-4 font-semibold">File Name</th>
                      <th className="text-left px-4 py-4 font-semibold hidden sm:table-cell">Type</th>
                      <th className="text-left px-4 py-4 font-semibold hidden md:table-cell">Folder</th>
                      <th className="text-right px-4 py-4 font-semibold hidden sm:table-cell">Size</th>
                      <th className="text-right px-6 py-4 font-semibold">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredItems.map((item) => (
                      <tr key={item.id} className="border-b border-[#273449]/50 hover:bg-white/[0.02] transition-colors">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-9 h-9 rounded-xl bg-[#050816] border border-[#273449] flex items-center justify-center flex-shrink-0">
                              {item.type === 'image' && <ImageIcon size={16} className="text-blue-500" />}
                              {item.type === 'video' && <Video size={16} className="text-cyan-500" />}
                              {item.type === 'document' && <FileText size={16} className="text-violet-500" />}
                            </div>
                            <div className="min-w-0">
                              <div className="text-xs font-semibold text-white truncate max-w-xs">{item.name}</div>
                              <div className="text-[10px] text-[#94A3B8] sm:hidden mt-0.5">{formatFileSize(item.size)}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-4 hidden sm:table-cell">
                          <span className="px-2 py-0.5 bg-[#050816] border border-[#273449]/50 rounded text-[10px] uppercase text-[#94A3B8] font-bold">
                            {item.type}
                          </span>
                        </td>
                        <td className="px-4 py-4 hidden md:table-cell">
                          <span className="text-xs text-[#94A3B8] flex items-center gap-1.5">
                            <Folder size={13} className="text-blue-400" /> {item.folder}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-right hidden sm:table-cell text-xs text-[#94A3B8] font-mono">
                          {formatFileSize(item.size)}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <div className="flex items-center justify-end gap-1.5">
                            <button
                              onClick={() => handleCopy(item.id, item.url)}
                              className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"
                              title="Copy URL"
                            >
                              {copiedId === item.id ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
                            </button>
                            <button
                              onClick={() => handleDelete(item.id)}
                              className="p-2 rounded-lg hover:bg-red-500/10 text-[#94A3B8] hover:text-red-400 transition-colors"
                              title="Delete"
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {filteredItems.length === 0 && (
            <div className="bg-[#0D1224] border border-[#273449] rounded-2xl py-16 text-center text-[#94A3B8] text-xs">
              No media items found in &ldquo;{activeFolder}&rdquo;. Upload files above!
            </div>
          )}
        </div>
      </div>

      {/* New Folder Modal */}
      {showFolderModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-md bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-[#273449] pb-4">
              <h3 className="font-display text-base font-bold text-white flex items-center gap-2">
                <FolderPlus size={18} className="text-blue-500" /> Create Media Folder
              </h3>
              <button onClick={() => setShowFolderModal(false)} className="text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold text-[#94A3B8]">Folder Name *</label>
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreateFolder()}
                placeholder="e.g. Client Logos, Case Study Assets..."
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                autoFocus
              />
            </div>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setShowFolderModal(false)}
                className="px-4 py-2 border border-[#273449] text-[#94A3B8] text-xs font-semibold rounded-xl hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateFolder}
                className="px-5 py-2 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-semibold rounded-xl hover:shadow-glow-sm"
              >
                Create Folder
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
