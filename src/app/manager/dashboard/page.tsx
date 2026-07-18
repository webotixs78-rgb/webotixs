'use client'

import React, { useState, useEffect } from 'react'
import { Plus, CheckCircle2, Clock, AlertCircle, Users, FolderKanban, Briefcase, CheckSquare, Upload, MessageSquare, ArrowUpRight, X, Save, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function ManagerDashboardPage() {
  const [projects, setProjects] = useState<any[]>([])
  const [tasks, setTasks] = useState<any[]>([])
  const [teamMembers, setTeamMembers] = useState<any[]>([])
  const [activeTab, setActiveTab] = useState<'projects' | 'tasks' | 'team'>('projects')
  const [isCreatingTask, setIsCreatingTask] = useState(false)
  const [newTask, setNewTask] = useState({
    title: '',
    project_id: '',
    description: '',
    role_required: 'UI/UX Designer',
    assigned_to: '',
    priority: 'medium',
    due_date: new Date().toISOString().split('T')[0],
    status: 'Todo',
  })

  useEffect(() => {
    async function loadData() {
      try {
        const pRes = await fetch('/api/crm/projects').catch(() => null)
        if (pRes && pRes.ok) {
          const pData = await pRes.json()
          if (pData.projects && pData.projects.length > 0) {
            setProjects(pData.projects)
            localStorage.setItem('webotixs_crm_projects', JSON.stringify(pData.projects))
          }
        }
      } catch (e) {}

      const localProjects = localStorage.getItem('webotixs_crm_projects')
      const localTasks = localStorage.getItem('webotixs_crm_tasks')
      const localTeam = localStorage.getItem('webotixs_team_members')

      if (localProjects) setProjects(JSON.parse(localProjects))
      if (localTasks) setTasks(JSON.parse(localTasks))
      if (localTeam) setTeamMembers(JSON.parse(localTeam))
    }
    loadData()
  }, [])

  const handleCreateTask = () => {
    if (!newTask.title || !newTask.project_id) {
      alert('Please select a project and enter a task title.')
      return
    }

    const targetProject = projects.find((p) => p.id === newTask.project_id)
    const createdTask = {
      id: `task-mgr-${Date.now()}`,
      project_id: newTask.project_id,
      project_title: targetProject ? targetProject.title : 'Assigned Project',
      step_order: tasks.length + 1,
      title: newTask.title,
      description: newTask.description || 'Assigned by Team Manager.',
      role_required: newTask.role_required,
      assigned_to: newTask.assigned_to || newTask.role_required,
      priority: newTask.priority,
      due_date: newTask.due_date,
      status: newTask.status,
    }

    const updatedTasks = [createdTask, ...tasks]
    setTasks(updatedTasks)
    localStorage.setItem('webotixs_crm_tasks', JSON.stringify(updatedTasks))
    setIsCreatingTask(false)
    setNewTask({
      title: '',
      project_id: projects[0]?.id || '',
      description: '',
      role_required: 'UI/UX Designer',
      assigned_to: '',
      priority: 'medium',
      due_date: new Date().toISOString().split('T')[0],
      status: 'Todo',
    })
    alert('✅ Task Created & Assigned! Team members assigned to this department will see it immediately on their board.')
  }

  const handleApproveWork = (taskId: string) => {
    const updated = tasks.map((t) => (t.id === taskId ? { ...t, status: 'Completed', approved_by_manager: true } : t))
    setTasks(updated)
    localStorage.setItem('webotixs_crm_tasks', JSON.stringify(updated))
    alert('✅ Work Approved! Task marked as Completed.')
  }

  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="bg-gradient-to-r from-purple-900/40 via-[#0D1224] to-blue-900/30 border border-purple-500/30 rounded-3xl p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xl">
        <div>
          <span className="px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 text-xs font-bold uppercase tracking-wider">
            SaaS Team Management Dashboard
          </span>
          <h1 className="font-display text-2xl md:text-3xl font-bold text-white tracking-tight mt-2">
            Manager Command Center
          </h1>
          <p className="text-xs md:text-sm text-[#94A3B8] max-w-xl mt-1">
            Oversee assigned client projects, delegate pipeline tasks to department staff, and review submitted deliverables.
          </p>
        </div>

        <button
          onClick={() => {
            if (projects.length === 0) {
              alert('No active projects found. Ask your Super Admin to assign or create a project first.')
              return
            }
            setNewTask((prev) => ({ ...prev, project_id: projects[0]?.id || '' }))
            setIsCreatingTask(true)
          }}
          className="flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-purple-600 to-blue-500 text-white font-semibold rounded-2xl shadow-lg hover:shadow-purple-500/25 transition-all text-xs shrink-0"
        >
          <Plus size={16} /> Create & Assign Task
        </button>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-3 border-b border-[#273449] pb-3">
        <button
          onClick={() => setActiveTab('projects')}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
            activeTab === 'projects' ? 'bg-purple-600 text-white shadow-md' : 'text-[#94A3B8] hover:bg-white/5 hover:text-white'
          )}
        >
          <FolderKanban size={14} /> Assigned Projects ({projects.length})
        </button>
        <button
          onClick={() => setActiveTab('tasks')}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
            activeTab === 'tasks' ? 'bg-purple-600 text-white shadow-md' : 'text-[#94A3B8] hover:bg-white/5 hover:text-white'
          )}
        >
          <CheckSquare size={14} /> Pipeline Tasks ({tasks.length})
        </button>
        <button
          onClick={() => setActiveTab('team')}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all',
            activeTab === 'team' ? 'bg-purple-600 text-white shadow-md' : 'text-[#94A3B8] hover:bg-white/5 hover:text-white'
          )}
        >
          <Users size={14} /> Assigned Team Members ({teamMembers.length})
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'projects' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project) => (
            <div key={project.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 space-y-4 hover:border-purple-500/40 transition-all shadow-md">
              <div className="flex items-start justify-between gap-3">
                <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase">
                  {project.package_type || 'Assigned Project'}
                </span>
                <span className={cn('px-2.5 py-1 rounded-full text-[10px] font-bold uppercase border', project.status === 'Completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-purple-500/10 text-purple-400 border-purple-500/30')}>
                  {project.status}
                </span>
              </div>

              <h3 className="font-display font-bold text-white text-base line-clamp-1">{project.title}</h3>
              <p className="text-xs text-[#94A3B8] line-clamp-2">{project.notes || 'No description provided.'}</p>

              {/* Progress Bar */}
              <div className="space-y-1.5 pt-2 border-t border-[#273449]/50">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#94A3B8] font-medium">Overall Completion</span>
                  <span className="text-white font-bold">{project.progress_percentage || 0}%</span>
                </div>
                <div className="h-2 w-full bg-[#050816] rounded-full overflow-hidden border border-[#273449]">
                  <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all" style={{ width: `${project.progress_percentage || 0}%` }} />
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 text-[11px] text-[#94A3B8]">
                <span>Client: <strong className="text-white">{project.client?.company_name || 'Assigned Client'}</strong></span>
                <span>Deadline: <strong className="text-purple-300">{project.deadline || 'TBD'}</strong></span>
              </div>
            </div>
          ))}
          {projects.length === 0 && (
            <div className="col-span-full bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
              <FolderKanban size={36} className="text-purple-400 mx-auto opacity-70" />
              <h3 className="font-display font-bold text-white text-base">No Assigned Projects Found</h3>
              <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                Once the Super Admin creates projects inside the Admin Panel and assigns them to your manager account, they will automatically populate here.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'tasks' && (
        <div className="space-y-4">
          {tasks.map((task) => (
            <div key={task.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 hover:border-purple-500/40 transition-all shadow">
              <div className="space-y-1">
                <div className="flex items-center gap-2.5">
                  <span className="px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-bold uppercase">
                    Role: {task.role_required}
                  </span>
                  <span className={cn('px-2 py-0.5 rounded text-[10px] font-bold uppercase border', task.status === 'Completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : task.status === 'Review' ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30')}>
                    {task.status}
                  </span>
                </div>
                <h4 className="font-display font-bold text-white text-sm">{task.title}</h4>
                <p className="text-xs text-[#94A3B8]">{task.description}</p>
                {task.deliverable_url && (
                  <div className="mt-2 text-xs">
                    <span className="text-[#94A3B8]">Submitted Deliverable: </span>
                    <a href={task.deliverable_url} target="_blank" rel="noreferrer" className="text-cyan-400 underline font-semibold flex items-center gap-1 inline-flex">
                      {task.deliverable_url} <ArrowUpRight size={12} />
                    </a>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-3 shrink-0">
                {task.status === 'Review' && !task.approved_by_manager && (
                  <button
                    onClick={() => handleApproveWork(task.id)}
                    className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 hover:bg-emerald-500/20 text-emerald-400 text-xs font-bold rounded-xl transition-all flex items-center gap-1.5"
                  >
                    <CheckCircle2 size={14} /> Approve Work
                  </button>
                )}
                <div className="text-right text-[11px] text-[#94A3B8]">
                  <div>Project: <strong className="text-white">{task.project_title || 'N/A'}</strong></div>
                  <div>Due: <strong className="text-purple-300">{task.due_date || 'TBD'}</strong></div>
                </div>
              </div>
            </div>
          ))}
          {tasks.length === 0 && (
            <div className="bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
              <CheckSquare size={36} className="text-purple-400 mx-auto opacity-70" />
              <h3 className="font-display font-bold text-white text-base">No Pipeline Tasks Created</h3>
              <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                Click &apos;Create & Assign Task&apos; above to delegate action items to your department team members.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'team' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {teamMembers.map((member) => (
            <div key={member.id} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 flex items-center gap-4 shadow-md">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-600/20 to-blue-600/20 border border-purple-500/30 flex items-center justify-center font-display font-bold text-lg text-purple-400 shrink-0">
                {member.name ? member.name.split(' ').map((n: any) => n[0]).join('') : 'TM'}
              </div>
              <div>
                <h3 className="font-display font-bold text-white text-sm">{member.name}</h3>
                <p className="text-xs text-purple-400 font-semibold">{member.position || member.role_department}</p>
                <p className="text-[11px] text-[#94A3B8] mt-1">{member.email}</p>
              </div>
            </div>
          ))}
          {teamMembers.length === 0 && (
            <div className="col-span-full bg-[#0D1224]/50 border border-dashed border-[#273449] rounded-3xl p-12 text-center space-y-3">
              <Users size={36} className="text-purple-400 mx-auto opacity-70" />
              <h3 className="font-display font-bold text-white text-base">No Staff Members Found</h3>
              <p className="text-[#94A3B8] text-xs max-w-md mx-auto">
                When the Super Admin generates Team Members and assigns their roles inside the Team Manager page, they will show up here.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Create Task Modal */}
      {isCreatingTask && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg bg-[#0D1224] border border-[#273449] rounded-3xl overflow-hidden shadow-2xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#273449]">
              <h3 className="font-display text-lg font-bold text-white">Create & Assign Department Task</h3>
              <button onClick={() => setIsCreatingTask(false)} className="p-2 rounded-lg text-[#94A3B8] hover:text-white">
                <X size={18} />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Select Project *</label>
                <select
                  value={newTask.project_id}
                  onChange={(e) => setNewTask({ ...newTask, project_id: e.target.value })}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500"
                >
                  {projects.map((p) => (
                    <option key={p.id} value={p.id} className="bg-[#0D1224] text-white">
                      {p.title}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Task Title *</label>
                <input
                  type="text"
                  value={newTask.title}
                  onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                  placeholder="e.g. Wireframe Mobile Responsive Views"
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Assign To Role / Department *</label>
                  <select
                    value={newTask.role_required}
                    onChange={(e) => setNewTask({ ...newTask, role_required: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500"
                  >
                    <option value="UI/UX Designer">UI/UX Designer</option>
                    <option value="Frontend Developer">Frontend Developer</option>
                    <option value="Backend Developer">Backend Developer</option>
                    <option value="WordPress Developer">WordPress Developer</option>
                    <option value="QA Tester">QA Tester</option>
                    <option value="SEO Specialist">SEO Specialist</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#94A3B8]">Due Date</label>
                  <input
                    type="date"
                    value={newTask.due_date}
                    onChange={(e) => setNewTask({ ...newTask, due_date: e.target.value })}
                    className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Task Instructions & Deliverable Expectations</label>
                <textarea
                  rows={3}
                  value={newTask.description}
                  onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                  placeholder="Provide detailed guidance for the assigned staff member..."
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-purple-500 resize-none"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#273449] bg-[#050816]/50">
              <button onClick={() => setIsCreatingTask(false)} className="px-5 py-2.5 border border-[#273449] text-[#94A3B8] text-sm font-semibold rounded-xl hover:text-white transition-colors">
                Cancel
              </button>
              <button onClick={handleCreateTask} className="px-6 py-2.5 bg-gradient-to-r from-purple-600 to-blue-500 text-white text-sm font-semibold rounded-xl shadow-lg hover:shadow-glow-sm transition-all">
                Assign & Spawn Task
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
