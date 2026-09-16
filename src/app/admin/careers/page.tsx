'use client'

import { useState, useEffect } from 'react'
import {
  Briefcase,
  Plus,
  Pencil,
  Trash2,
  Search,
  Check,
  X,
  FileText,
  User,
  Mail,
  Phone,
  Calendar,
  ExternalLink,
  MapPin,
  DollarSign,
  Clock,
  Eye,
  Loader2,
  Save,
} from 'lucide-react'
import { cn, formatDateShort } from '@/lib/utils'
import { getCMSData, saveCMSItem, deleteCMSItem } from '@/lib/data/cms'

interface JobPosition {
  id: string
  title: string
  department: string
  location: string
  type: 'Full-Time' | 'Contract' | 'Remote'
  salary_range: string
  description: string
  requirements: string[]
  status: 'published' | 'closed'
  created_at: string
}

interface JobApplication {
  id: string
  job_id: string
  job_title: string
  applicant_name: string
  email: string
  phone: string
  linkedin_url: string
  portfolio_url: string
  resume_url: string
  cover_letter: string
  status: 'new' | 'reviewing' | 'shortlisted' | 'rejected' | 'hired'
  applied_at: string
}

const mockPositions: JobPosition[] = [
  {
    id: 'job-1',
    title: 'Senior Full-Stack Engineer (Next.js / Node)',
    department: 'Engineering',
    location: 'Remote Worldwide',
    type: 'Full-Time',
    salary_range: '$110,000 - $145,000 USD / yr',
    description: 'We are seeking an experienced Full-Stack Engineer to architect enterprise web applications and headless e-commerce platforms using Next.js 16, TypeScript, and Supabase.',
    requirements: ['5+ years full-stack experience', 'Expertise in Next.js App Router & Server Actions', 'Strong SQL & PostgreSQL knowledge', 'Experience with cloud infrastructure (Vercel / AWS)'],
    status: 'published',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(),
  },
  {
    id: 'job-2',
    title: 'Lead UI/UX Designer & Design Systems Architect',
    department: 'Design',
    location: 'Dubai, UAE · Hybrid',
    type: 'Full-Time',
    salary_range: '$95,000 - $130,000 USD / yr',
    description: 'Create award-winning digital experiences, interactive prototypes, and comprehensive design tokens for international luxury and enterprise brands.',
    requirements: ['4+ years product design experience', 'Mastery of Figma, variables, and auto-layout', 'Experience with motion (GSAP / Framer Motion)', 'Portfolio showcasing high-end web applications'],
    status: 'published',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString(),
  },
]

const mockApplications: JobApplication[] = [
  {
    id: 'app-1',
    job_id: 'job-1',
    job_title: 'Senior Full-Stack Engineer (Next.js / Node)',
    applicant_name: 'David K. Vance',
    email: 'd.vance.dev@gmail.com',
    phone: '+1-415-892-3401',
    linkedin_url: 'https://linkedin.com/in/davidvance-dev',
    portfolio_url: 'https://davidvance.io',
    resume_url: '#',
    cover_letter: 'I have spent the last 6 years building high-concurrency fintech platforms using Next.js and Supabase. Webotixs’ engineering culture and design philosophy resonate deeply with my standards.',
    status: 'shortlisted',
    applied_at: new Date(Date.now() - 1000 * 60 * 60 * 18).toISOString(),
  },
  {
    id: 'app-2',
    job_id: 'job-2',
    job_title: 'Lead UI/UX Designer & Design Systems Architect',
    applicant_name: 'Elena Rostova',
    email: 'elena.design@rostova.co',
    phone: '+971-50-842-1902',
    linkedin_url: 'https://linkedin.com/in/elenarostova',
    portfolio_url: 'https://elenarostova.design',
    resume_url: '#',
    cover_letter: 'Currently based in Dubai and designing enterprise SaaS interfaces. I love building design tokens that bridge the gap perfectly with Tailwind CSS.',
    status: 'new',
    applied_at: new Date(Date.now() - 1000 * 60 * 60 * 36).toISOString(),
  },
  {
    id: 'app-3',
    job_id: 'job-1',
    job_title: 'Senior Full-Stack Engineer (Next.js / Node)',
    applicant_name: 'Omar Al-Jundi',
    email: 'omar.jundi@outlook.com',
    phone: '+962-79-123-4567',
    linkedin_url: 'https://linkedin.com/in/omaraljundi',
    portfolio_url: '',
    resume_url: '#',
    cover_letter: 'Experienced backend engineer looking to transition more into full-stack architecture.',
    status: 'reviewing',
    applied_at: new Date(Date.now() - 1000 * 60 * 60 * 72).toISOString(),
  },
]

const statusColors: Record<JobApplication['status'], string> = {
  new: 'bg-blue-500/10 text-blue-400 border-blue-500/25',
  reviewing: 'bg-amber-500/10 text-amber-400 border-amber-500/25',
  shortlisted: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25',
  rejected: 'bg-red-500/10 text-red-400 border-red-500/25',
  hired: 'bg-violet-500/10 text-violet-400 border-violet-500/25',
}

const emptyJob: Partial<JobPosition> = {
  title: '',
  department: 'Engineering',
  location: 'Remote Worldwide',
  type: 'Full-Time',
  salary_range: '',
  description: '',
  requirements: [],
  status: 'published',
}

export default function AdminCareersPage() {
  const [activeTab, setActiveTab] = useState<'positions' | 'applications'>('applications')
  const [positions, setPositions] = useState<JobPosition[]>([...mockPositions])
  const [applications, setApplications] = useState<JobApplication[]>([...mockApplications])
  const [search, setSearch] = useState('')
  const [editingJob, setEditingJob] = useState<Partial<JobPosition> | null>(null)
  const [isNewJob, setIsNewJob] = useState(false)
  const [reqInput, setReqInput] = useState('')
  const [viewingApp, setViewingApp] = useState<JobApplication | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getCMSData<JobPosition[]>('careers').then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        setPositions(data)
      }
    })
  }, [])

  const filteredApps = applications.filter(
    (a) =>
      a.applicant_name.toLowerCase().includes(search.toLowerCase()) ||
      a.job_title.toLowerCase().includes(search.toLowerCase()) ||
      a.email.toLowerCase().includes(search.toLowerCase())
  )

  const filteredPositions = positions.filter((p) => p.title.toLowerCase().includes(search.toLowerCase()))

  const handleUpdateAppStatus = (id: string, status: JobApplication['status']) => {
    setApplications((prev) => prev.map((a) => (a.id === id ? { ...a, status } : a)))
    if (viewingApp?.id === id) setViewingApp((prev) => (prev ? { ...prev, status } : null))
  }

  const handleDeleteJob = async (id: string) => {
    if (!window.confirm('Delete this position?')) return
    setPositions((prev) => prev.filter((p) => p.id !== id))
    await deleteCMSItem('careers', id)
  }

  const toggleJobStatus = async (id: string) => {
    const target = positions.find((p) => p.id === id)
    if (!target) return
    const updatedJob = { ...target, status: target.status === 'published' ? 'closed' : 'published' } as JobPosition
    setPositions((prev) => prev.map((p) => (p.id === id ? updatedJob : p)))
    await saveCMSItem('careers', updatedJob)
  }

  const openNewJob = () => {
    setEditingJob({ ...emptyJob })
    setIsNewJob(true)
  }
  const openEditJob = (p: JobPosition) => {
    setEditingJob({ ...p })
    setIsNewJob(false)
  }

  const handleSaveJob = async () => {
    if (!editingJob?.title || !editingJob?.description) return
    setSaving(true)
    const now = new Date().toISOString()
    if (isNewJob) {
      const newJob: JobPosition = {
        ...(editingJob as JobPosition),
        id: `job-${Date.now()}`,
        requirements: editingJob.requirements || [],
        created_at: now,
      }
      setPositions((prev) => [...prev, newJob])
      await saveCMSItem('careers', newJob)
    } else {
      const updatedJob = { ...editingJob } as JobPosition
      setPositions((prev) => prev.map((p) => (p.id === editingJob.id ? updatedJob : p)))
      await saveCMSItem('careers', updatedJob)
    }
    setSaving(false)
    setEditingJob(null)
  }

  const addReq = () => {
    if (!reqInput.trim() || !editingJob) return
    setEditingJob({ ...editingJob, requirements: [...(editingJob.requirements || []), reqInput.trim()] })
    setReqInput('')
  }
  const removeReq = (idx: number) => {
    if (!editingJob) return
    const r = [...(editingJob.requirements || [])]
    r.splice(idx, 1)
    setEditingJob({ ...editingJob, requirements: r })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2.5">
            <Briefcase size={24} className="text-blue-500" /> Careers & Job Applications
          </h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Manage open job positions and evaluate candidate submissions.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="bg-[#0D1224] border border-[#273449] p-1 rounded-xl flex gap-1">
            <button
              onClick={() => setActiveTab('applications')}
              className={cn(
                'px-4 py-2 rounded-lg text-xs font-semibold transition-all flex items-center gap-2',
                activeTab === 'applications'
                  ? 'bg-blue-600 text-white shadow-glow-sm'
                  : 'text-[#94A3B8] hover:text-white'
              )}
            >
              <span>Applications</span>
              <span className="px-1.5 py-0.5 rounded-full bg-black/30 text-[10px] font-bold">
                {applications.length}
              </span>
            </button>
            <button
              onClick={() => setActiveTab('positions')}
              className={cn(
                'px-4 py-2 rounded-lg text-xs font-semibold transition-all flex items-center gap-2',
                activeTab === 'positions' ? 'bg-blue-600 text-white shadow-glow-sm' : 'text-[#94A3B8] hover:text-white'
              )}
            >
              <span>Job Openings</span>
              <span className="px-1.5 py-0.5 rounded-full bg-black/30 text-[10px] font-bold">
                {positions.length}
              </span>
            </button>
          </div>

          {activeTab === 'positions' && (
            <button
              onClick={openNewJob}
              className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-semibold rounded-xl hover:shadow-glow-sm transition-all"
            >
              <Plus size={16} /> Add Position
            </button>
          )}
        </div>
      </div>

      {/* Search Bar */}
      <div className="relative max-w-md">
        <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#94A3B8]/50" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={activeTab === 'applications' ? 'Search candidates, roles, email...' : 'Search job titles...'}
          className="w-full pl-11 pr-4 py-2.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm placeholder:text-[#94A3B8]/30 focus:outline-none focus:border-blue-500/50"
        />
      </div>

      {/* TAB 1: APPLICATIONS */}
      {activeTab === 'applications' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#273449] text-[#94A3B8] text-xs uppercase tracking-wider bg-[#0A0E1F]">
                  <th className="text-left px-6 py-4 font-semibold">Candidate</th>
                  <th className="text-left px-4 py-4 font-semibold hidden md:table-cell">Job Position</th>
                  <th className="text-center px-4 py-4 font-semibold">Status</th>
                  <th className="text-center px-4 py-4 font-semibold hidden lg:table-cell">Applied Date</th>
                  <th className="text-right px-6 py-4 font-semibold">Review</th>
                </tr>
              </thead>
              <tbody>
                {filteredApps.map((app) => (
                  <tr key={app.id} className="border-b border-[#273449]/50 hover:bg-white/[0.02] transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-500 text-xs font-bold">
                          {app.applicant_name[0]}
                        </div>
                        <div>
                          <div className="font-semibold text-white text-sm">{app.applicant_name}</div>
                          <div className="text-[10px] text-[#94A3B8]">{app.email}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4 hidden md:table-cell">
                      <div className="text-xs font-medium text-white max-w-xs truncate">{app.job_title}</div>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <span
                        className={cn(
                          'px-2.5 py-1 rounded-full text-[9px] font-bold uppercase border',
                          statusColors[app.status]
                        )}
                      >
                        {app.status}
                      </span>
                    </td>
                    <td className="px-4 py-4 text-center hidden lg:table-cell text-xs text-[#94A3B8]">
                      {formatDateShort(app.applied_at)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={() => setViewingApp(app)}
                        className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"
                        title="Review Application"
                      >
                        <Eye size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredApps.length === 0 && (
            <div className="py-16 text-center text-[#94A3B8] text-xs">No candidate applications found.</div>
          )}
        </div>
      )}

      {/* TAB 2: JOB POSITIONS */}
      {activeTab === 'positions' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {filteredPositions.map((job) => (
            <div
              key={job.id}
              className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 space-y-4 flex flex-col justify-between hover:border-blue-500/40 transition-colors"
            >
              <div className="space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="text-[10px] text-blue-400 font-bold uppercase tracking-wider">
                      {job.department} · {job.type}
                    </span>
                    <h3 className="font-display text-base font-bold text-white mt-1">{job.title}</h3>
                  </div>
                  <button
                    onClick={() => toggleJobStatus(job.id)}
                    className={cn(
                      'px-2 py-0.5 rounded-full text-[8px] font-bold uppercase border flex-shrink-0',
                      job.status === 'published'
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25'
                        : 'bg-amber-500/10 text-amber-400 border-amber-500/25'
                    )}
                  >
                    {job.status}
                  </button>
                </div>

                <div className="flex items-center gap-4 text-xs text-[#94A3B8]">
                  <span className="flex items-center gap-1">
                    <MapPin size={12} className="text-blue-500" /> {job.location}
                  </span>
                  <span className="flex items-center gap-1 font-semibold text-emerald-400">
                    <DollarSign size={12} /> {job.salary_range}
                  </span>
                </div>

                <p className="text-xs text-[#94A3B8] leading-relaxed line-clamp-3 pt-1">{job.description}</p>

                <div className="flex flex-wrap gap-1.5 pt-2">
                  {job.requirements.slice(0, 3).map((r, i) => (
                    <span key={i} className="px-2 py-0.5 bg-[#050816] border border-[#273449] rounded text-[10px] text-[#94A3B8]">
                      ✓ {r}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-[#273449]/40">
                <span className="text-[10px] text-[#94A3B8]/60">Posted {formatDateShort(job.created_at)}</span>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => openEditJob(job)}
                    className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white transition-colors"
                  >
                    <Pencil size={14} />
                  </button>
                  <button
                    onClick={() => handleDeleteJob(job.id)}
                    className="p-2 rounded-lg hover:bg-red-500/10 text-[#94A3B8] hover:text-red-400 transition-colors"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Candidate Detail Modal */}
      {viewingApp && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-xl max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">Candidate Application</h2>
              <button onClick={() => setViewingApp(null)} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              {/* Profile Header */}
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
                    <User size={16} className="text-blue-500" /> {viewingApp.applicant_name}
                  </h3>
                  <p className="text-xs text-blue-400 font-semibold mt-0.5">Applied for: {viewingApp.job_title}</p>
                </div>
                <span className={cn('px-3 py-1 rounded-full text-xs font-bold uppercase border', statusColors[viewingApp.status])}>
                  {viewingApp.status}
                </span>
              </div>

              {/* Contact Links */}
              <div className="grid grid-cols-2 gap-4 bg-[#050816] border border-[#273449]/50 p-4 rounded-2xl text-xs">
                <a href={`mailto:${viewingApp.email}`} className="text-blue-400 hover:underline flex items-center gap-2">
                  <Mail size={13} /> {viewingApp.email}
                </a>
                <div className="text-white flex items-center gap-2">
                  <Phone size={13} className="text-cyan-500" /> {viewingApp.phone || 'N/A'}
                </div>
                {viewingApp.linkedin_url && (
                  <a href={viewingApp.linkedin_url} target="_blank" rel="noreferrer" className="text-blue-400 hover:underline flex items-center gap-2">
                    <ExternalLink size={13} /> LinkedIn Profile
                  </a>
                )}
                {viewingApp.portfolio_url && (
                  <a href={viewingApp.portfolio_url} target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline flex items-center gap-2">
                    <ExternalLink size={13} /> Portfolio URL
                  </a>
                )}
              </div>

              {/* Cover Letter */}
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold uppercase tracking-wider text-[#94A3B8]">Cover Letter / Notes</label>
                <div className="p-4 bg-[#050816] border border-[#273449] rounded-2xl text-white text-xs leading-relaxed">
                  {viewingApp.cover_letter || 'No cover letter provided.'}
                </div>
              </div>

              {/* Status Update Pipeline */}
              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-wider text-[#94A3B8]">Update Candidate Stage</label>
                <div className="flex flex-wrap gap-2">
                  {(['new', 'reviewing', 'shortlisted', 'rejected', 'hired'] as const).map((status) => (
                    <button
                      key={status}
                      onClick={() => handleUpdateAppStatus(viewingApp.id, status)}
                      className={cn(
                        'px-4 py-2 rounded-xl text-xs font-bold uppercase border transition-all',
                        viewingApp.status === status
                          ? 'bg-blue-600 text-white border-blue-600 shadow-glow-sm'
                          : 'bg-[#050816] text-[#94A3B8] border-[#273449] hover:border-blue-500/40'
                      )}
                    >
                      {status}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Job Position Modal Editor */}
      {editingJob && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-2xl max-h-[90vh] bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h2 className="font-display text-lg font-bold text-white">{isNewJob ? 'Create Job Opening' : 'Edit Position'}</h2>
              <button onClick={() => setEditingJob(null)} className="p-2 rounded-lg hover:bg-white/5 text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Job Title *</label>
                  <input
                    type="text"
                    value={editingJob.title || ''}
                    onChange={(e) => setEditingJob({ ...editingJob, title: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Department *</label>
                  <input
                    type="text"
                    value={editingJob.department || ''}
                    onChange={(e) => setEditingJob({ ...editingJob, department: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Location</label>
                  <input
                    type="text"
                    value={editingJob.location || ''}
                    onChange={(e) => setEditingJob({ ...editingJob, location: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Job Type</label>
                  <select
                    value={editingJob.type || 'Full-Time'}
                    onChange={(e) => setEditingJob({ ...editingJob, type: e.target.value as any })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  >
                    <option value="Full-Time">Full-Time</option>
                    <option value="Contract">Contract</option>
                    <option value="Remote">Remote</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Salary Range</label>
                  <input
                    type="text"
                    value={editingJob.salary_range || ''}
                    onChange={(e) => setEditingJob({ ...editingJob, salary_range: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Description *</label>
                <textarea
                  rows={4}
                  value={editingJob.description || ''}
                  onChange={(e) => setEditingJob({ ...editingJob, description: e.target.value })}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none"
                />
              </div>

              {/* Requirements */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-[#94A3B8]">Requirements / Key Skills</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={reqInput}
                    onChange={(e) => setReqInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addReq())}
                    className="flex-1 px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                    placeholder="Add requirement..."
                  />
                  <button
                    onClick={addReq}
                    className="px-4 py-2 bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded-xl"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  {(editingJob.requirements || []).map((r, i) => (
                    <span key={i} className="flex items-center gap-1.5 px-3 py-1 bg-[#050816] border border-[#273449] rounded-lg text-xs text-[#94A3B8]">
                      ✓ {r}
                      <button onClick={() => removeReq(i)} className="text-red-400">
                        <X size={10} />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449]">
              <button
                onClick={() => setEditingJob(null)}
                className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveJob}
                disabled={saving}
                className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-sm font-semibold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50"
              >
                {saving ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> Saving...
                  </>
                ) : (
                  <>
                    <Save size={14} /> {isNewJob ? 'Create Position' : 'Save Changes'}
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
