import Link from 'next/link'
import {
  Briefcase,
  FolderKanban,
  FileText,
  Users,
  MessageSquareCode,
  ArrowRight,
  TrendingUp,
  Mail,
  User,
  Clock,
} from 'lucide-react'
import { mockServices, mockPortfolio, mockTeam, mockBlogPosts } from '@/lib/data/mock'

export const metadata = {
  title: 'Dashboard Overview',
  description: 'Webotixs admin panel overview cards and recent metrics.',
}

export default function AdminDashboardPage() {
  const serviceCount = mockServices.length
  const projectCount = mockPortfolio.length
  const teamCount = mockTeam.length
  const blogCount = mockBlogPosts.length

  const stats = [
    { label: 'Total Services', value: serviceCount, icon: Briefcase, color: 'text-blue-500 bg-blue-500/10 border-blue-500/20' },
    { label: 'Active Projects', value: projectCount, icon: FolderKanban, color: 'text-cyan-500 bg-cyan-500/10 border-cyan-500/20' },
    { label: 'Team Members', value: teamCount, icon: Users, color: 'text-violet-500 bg-violet-500/10 border-violet-500/20' },
    { label: 'Published Blogs', value: blogCount, icon: FileText, color: 'text-amber-500 bg-amber-500/10 border-amber-500/20' },
  ]

  // Mock contact inquiries
  const recentInquiries = [
    { id: '1', name: 'Al-Khaleej Retail', email: 'contact@alkhaleej.ae', service: 'E-Commerce Solutions', priority: 'high', date: 'Just Now' },
    { id: '2', name: 'Robert Finch', email: 'r.finch@finchinvest.com', service: 'Web Design & Development', priority: 'medium', date: '2 Hours Ago' },
    { id: '3', name: 'EduLearn Inc', email: 'support@edulearn.org', service: 'Cloud & DevOps', priority: 'low', date: '1 Day Ago' },
  ]

  return (
    <div className="space-y-8">
      {/* Welcome Banner */}
      <div>
        <h1 className="font-display text-3xl font-bold text-white">Dashboard Overview</h1>
        <p className="text-[#94A3B8] text-sm mt-1">Quick metrics overview and platform state summary.</p>
      </div>

      {/* Grid Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.label} className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6 flex items-center justify-between">
              <div>
                <span className="text-[#94A3B8] text-xs font-semibold uppercase tracking-wider block mb-1">
                  {stat.label}
                </span>
                <span className="font-display text-3xl font-bold text-white">{stat.value}</span>
              </div>
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${stat.color}`}>
                <Icon size={20} />
              </div>
            </div>
          )}
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* CRM Quick Look */}
        <div className="lg:col-span-2 bg-[#0D1224] border border-[#273449] rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-[#273449]">
            <h2 className="font-display text-lg font-bold text-white flex items-center gap-2">
              <MessageSquareCode size={18} className="text-blue-500" />
              Recent Inquiries
            </h2>
            <Link href="/admin/crm" className="text-xs text-blue-500 hover:underline flex items-center gap-1">
              View Pipeline <ArrowRight size={12} />
            </Link>
          </div>

          <div className="space-y-4">
            {recentInquiries.map((inq) => (
              <div key={inq.id} className="p-4 bg-[#050816]/60 border border-[#273449]/50 rounded-xl flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-500">
                    <User size={16} />
                  </div>
                  <div>
                    <div className="font-display text-sm font-semibold text-white">{inq.name}</div>
                    <div className="text-[10px] text-[#94A3B8]">{inq.email} &middot; {inq.service}</div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  {/* Priority Tag */}
                  <span className={`px-2.5 py-0.5 rounded-full text-[9px] font-bold uppercase ${
                    inq.priority === 'high' ? 'bg-red-500/10 text-red-400 border border-red-500/25' :
                    inq.priority === 'medium' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/25' :
                    'bg-slate-500/10 text-slate-400 border border-slate-500/25'
                  }`}>
                    {inq.priority}
                  </span>
                  <span className="text-[10px] text-[#94A3B8] flex items-center gap-1">
                    <Clock size={10} />
                    {inq.date}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-[#0D1224] border border-[#273449] rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-[#273449]">
            <h2 className="font-display text-lg font-bold text-white flex items-center gap-2">
              <TrendingUp size={18} className="text-cyan-500" />
              Quick Actions
            </h2>
          </div>

          <div className="space-y-3">
            {[
              { label: 'Add New Service', href: '/admin/services' },
              { label: 'Add Portfolio Project', href: '/admin/portfolio' },
              { label: 'Draft New Blog', href: '/admin/blogs' },
              { label: 'Register Team Member', href: '/admin/team' },
            ].map((act) => (
              <Link
                key={act.label}
                href={act.href}
                className="flex items-center justify-between px-4 py-3 bg-[#050816]/40 border border-[#273449]/40 hover:border-blue-500/40 rounded-xl text-xs font-semibold hover:text-blue-500 transition-colors group"
              >
                <span>{act.label}</span>
                <ArrowRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
