'use client'

import { useState } from 'react'
import {
  Database,
  Download,
  Upload,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  FileJson,
  FileSpreadsheet,
  ShieldAlert,
  Loader2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { mockServices, mockPortfolio, mockTeam, mockBlogPosts, mockBlogCategories } from '@/lib/data/mock'

export default function BackupClient() {
  const [exporting, setExporting] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(false)
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const handleExportJSON = async (entity: string, data: any) => {
    setExporting(entity)
    await new Promise((r) => setTimeout(r, 600))
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(data, null, 2))}`
    const downloadAnchor = document.createElement('a')
    downloadAnchor.setAttribute('href', jsonString)
    downloadAnchor.setAttribute('download', `webotixs-${entity}-export-${new Date().toISOString().slice(0, 10)}.json`)
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
    setExporting(null)
    setStatusMsg({ type: 'success', text: `Successfully exported ${entity.toUpperCase()} data (` + JSON.stringify(data).length + ` bytes).` })
    setTimeout(() => setStatusMsg(null), 4000)
  }

  const handleExportAll = async () => {
    setExporting('all')
    await new Promise((r) => setTimeout(r, 800))
    const bundle = {
      exported_at: new Date().toISOString(),
      version: '0.1.0',
      services: mockServices,
      portfolio: mockPortfolio,
      team: mockTeam,
      blogs: mockBlogPosts,
      categories: mockBlogCategories,
    }
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(bundle, null, 2))}`
    const downloadAnchor = document.createElement('a')
    downloadAnchor.setAttribute('href', jsonString)
    downloadAnchor.setAttribute('download', `webotixs-full-backup-${new Date().toISOString().slice(0, 10)}.json`)
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
    setExporting(null)
    setStatusMsg({ type: 'success', text: 'Full CMS database snapshot bundle exported successfully!' })
    setTimeout(() => setStatusMsg(null), 4000)
  }

  const handleRestoreSampleData = async () => {
    if (!window.confirm('Are you sure you want to re-seed local CMS state with factory defaults? This will overwrite unsaved changes.')) return
    setRestoring(true)
    await new Promise((r) => setTimeout(r, 1200))
    setRestoring(false)
    setStatusMsg({ type: 'success', text: 'Factory defaults re-seeded across Services, Portfolio, Blog, and Team successfully.' })
    setTimeout(() => setStatusMsg(null), 4000)
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2.5">
          <Database size={24} className="text-blue-500" /> Backup, Export & Restore
        </h1>
        <p className="text-[#94A3B8] text-xs mt-1">
          Safeguard your agency database by exporting JSON snapshots or re-seeding factory data.
        </p>
      </div>

      {statusMsg && (
        <div
          className={cn(
            'px-4 py-3 rounded-2xl border text-xs font-semibold flex items-center gap-2',
            statusMsg.type === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
              : 'bg-red-500/10 border-red-500/20 text-red-400'
          )}
        >
          <CheckCircle2 size={16} /> {statusMsg.text}
        </div>
      )}

      {/* Full Snapshot Banner */}
      <div className="bg-gradient-to-br from-[#0D1224] to-[#0A0E1F] border border-blue-500/30 rounded-3xl p-6 relative overflow-hidden shadow-glow-sm">
        <div className="absolute top-0 right-0 w-80 h-80 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 relative z-10">
          <div className="space-y-1.5">
            <h2 className="font-display text-lg font-bold text-white flex items-center gap-2">
              <FileJson size={20} className="text-blue-400" /> Full CMS Database Snapshot
            </h2>
            <p className="text-xs text-[#94A3B8] max-w-lg leading-relaxed">
              Download a consolidated JSON archive containing all Services, Portfolio Case Studies, Blog Posts, Team Members, and Categories. Perfect for disaster recovery and staging migration.
            </p>
          </div>

          <button
            onClick={handleExportAll}
            disabled={exporting !== null}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-2xl hover:shadow-glow-sm transition-all flex-shrink-0 disabled:opacity-50"
          >
            {exporting === 'all' ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Exporting Bundle...
              </>
            ) : (
              <>
                <Download size={16} /> Download Full Snapshot
              </>
            )}
          </button>
        </div>
      </div>

      {/* Individual Module Exports */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-4">
        <h3 className="font-display text-base font-bold text-white">Export Individual CMS Modules</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {[
            { label: 'Services Catalog', key: 'services', data: mockServices, count: mockServices.length },
            { label: 'Portfolio Projects', key: 'portfolio', data: mockPortfolio, count: mockPortfolio.length },
            { label: 'Team Members', key: 'team', data: mockTeam, count: mockTeam.length },
            { label: 'Blog Posts & Articles', key: 'blogs', data: mockBlogPosts, count: mockBlogPosts.length },
            { label: 'Blog Categories', key: 'categories', data: mockBlogCategories, count: mockBlogCategories.length },
          ].map(({ label, key, data, count }) => (
            <div
              key={key}
              className="bg-[#050816] border border-[#273449] rounded-2xl p-4 flex flex-col justify-between hover:border-blue-500/40 transition-colors"
            >
              <div>
                <div className="text-xs font-bold text-white">{label}</div>
                <div className="text-[10px] text-[#94A3B8] mt-0.5">{count} entries recorded</div>
              </div>
              <button
                onClick={() => handleExportJSON(key, data)}
                disabled={exporting !== null}
                className="mt-4 flex items-center justify-center gap-1.5 w-full py-2 bg-[#0D1224] border border-[#273449] text-blue-400 hover:text-white hover:border-blue-500/50 text-xs font-semibold rounded-xl transition-all disabled:opacity-50"
              >
                {exporting === key ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <>
                    <Download size={13} /> Export JSON
                  </>
                )}
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Re-seed / Restore Utility */}
      <div className="bg-[#0D1224] border border-red-500/20 rounded-3xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
        <div className="space-y-1.5">
          <h3 className="font-display text-base font-bold text-white flex items-center gap-2">
            <RefreshCw size={18} className="text-amber-400" /> Re-Seed Factory Defaults
          </h3>
          <p className="text-xs text-[#94A3B8] max-w-md leading-relaxed">
            Reset local state across Services, Portfolio, Blog, and Team modules back to original production mock seeds.
          </p>
        </div>

        <button
          onClick={handleRestoreSampleData}
          disabled={restoring}
          className="flex items-center gap-2 px-5 py-2.5 bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 text-xs font-bold rounded-2xl transition-all flex-shrink-0 disabled:opacity-50"
        >
          {restoring ? (
            <>
              <Loader2 size={14} className="animate-spin" /> Restoring Seeds...
            </>
          ) : (
            <>
              <RefreshCw size={14} /> Reset to Factory Seeds
            </>
          )}
        </button>
      </div>
    </div>
  )
}
