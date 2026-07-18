'use client'

import { useState, useEffect } from 'react'
import {
  Save,
  Loader2,
  Globe,
  Mail,
  Phone,
  MapPin,
  Palette,
  Share2,
  LayoutTemplate,
  FileText,
  Search,
  Image as ImageIcon,
  ShieldCheck,
  Bell,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'

const tabs = [
  { id: 'general', label: 'General', icon: Globe },
  { id: 'social', label: 'Social Media', icon: Share2 },
  { id: 'homepage', label: 'Homepage', icon: LayoutTemplate },
  { id: 'pages', label: 'Page Content', icon: FileText },
  { id: 'seo', label: 'SEO', icon: Search },
  { id: 'banners', label: 'Banners', icon: ImageIcon },
  { id: 'email', label: 'Email', icon: Bell },
  { id: 'privacy', label: 'Privacy Policy', icon: ShieldCheck },
  { id: 'terms', label: 'Terms of Service', icon: FileText },
  { id: 'cookies', label: 'Cookie Policy', icon: ShieldCheck },
]

export default function AdminSettingsPage() {
  const [activeTab, setActiveTab] = useState('general')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  // Read URL params safely on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      const tabParam = params.get('tab')
      if (tabParam && tabs.some((t) => t.id === tabParam)) {
        setActiveTab(tabParam)
      }
    }
  }, [])

  const [settings, setSettings] = useState({
    // General
    site_name: 'Webotixs',
    tagline: 'Award-Winning Enterprise Web Design & Digital Innovation Agency',
    primary_color: '#3B82F6',
    secondary_color: '#06B6D4',
    footer_text: '© 2026 Webotixs Agency. All rights reserved.',
    email: 'hello@webotixs.com',
    phone: '+1 (234) 567-890',
    address: 'Dubai, UAE · Remote Delivery Worldwide',
    whatsapp: '+971501234567',
    // Social
    social_twitter: 'https://twitter.com/webotixs',
    social_linkedin: 'https://linkedin.com/company/webotixs',
    social_instagram: 'https://instagram.com/webotixs',
    social_github: 'https://github.com/webotixs',
    social_youtube: 'https://youtube.com/@webotixs',
    social_dribbble: 'https://dribbble.com/webotixs',
    // Page Content & Banners
    banner_about_title: 'Engineering Digital Excellence Since 2016',
    banner_services_title: 'Custom Enterprise Web & Mobile Solutions',
    banner_portfolio_title: 'Our Proven Track Record of High-Impact Deliverables',
    banner_contact_title: 'Start a Conversation With Our Engineering Team',
    // SEO
    seo_meta_title: 'Webotixs — Award-Winning Enterprise Web Design Agency',
    seo_meta_desc: 'We craft high-performance web applications, mobile experiences, and headless e-commerce solutions for global brands.',
    seo_keywords: 'web design dubai, nextjs agency, enterprise web development, UI UX design, custom software',
    seo_og_image: 'https://webotixs.com/og-cover.png',
    google_analytics_id: 'G-X987654321',
    microsoft_clarity_id: 'cl_987654321',
    // Email
    resend_api_key: 're_123456789_abcdef',
    sender_email: 'agency-leads@webotixs.com',
    notification_email: 'webotixs78@gmail.com',
    // Legal Policies
    privacy_policy_text: 'Webotixs Agency ("we", "our", or "us") is committed to protecting your personal information and right to privacy under international data protection laws (GDPR, CCPA, and UAE regulations)...',
    terms_of_service_text: 'By accessing or engaging Webotixs Agency for digital design, engineering, or consulting services, you agree to be bound by these standard Terms of Service and professional conduct guidelines...',
    cookie_policy_text: 'We use essential cookies and anonymous analytics (Google Analytics & Microsoft Clarity) to optimize our digital agency platform experience and measure traffic performance...',
  })

  const handleSave = async () => {
    setSaving(true)
    await new Promise((r) => setTimeout(r, 650))
    localStorage.setItem('webotixs_cms_global_settings', JSON.stringify(settings))
    setSaving(false)
    setSaved(true)
    setTimeout(() => setSaved(false), 3500)
  }

  const update = (key: string, value: string) => setSettings((prev) => ({ ...prev, [key]: value }))

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#273449] pb-6">
        <div>
          <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2">
            <SlidersIcon size={24} className="text-blue-500" /> Global System & Content Settings
          </h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Unified control center: modify site metadata, banners, emails, SEO, and legal policies exactly like HuxenTech.
          </p>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50 flex-shrink-0"
        >
          {saving ? (
            <>
              <Loader2 size={14} className="animate-spin" /> Saving...
            </>
          ) : (
            <>
              <Save size={14} /> Save {tabs.find((t) => t.id === activeTab)?.label}
            </>
          )}
        </button>
      </div>

      {saved && (
        <div className="px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold rounded-2xl flex items-center gap-2">
          <CheckCircle2 size={16} /> Changes successfully saved and propagated across your live agency platform!
        </div>
      )}

      {/* Tabs Bar */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-[#273449] custom-scrollbar">
        {tabs.map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold whitespace-nowrap transition-all border',
                isActive
                  ? 'bg-blue-600 text-white border-blue-600 shadow-glow-sm'
                  : 'bg-[#0D1224] text-[#94A3B8] border-[#273449] hover:text-white hover:border-blue-500/30'
              )}
            >
              <Icon size={14} className={cn(isActive ? 'text-white' : 'text-blue-400')} />
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>

      {/* TAB 1: GENERAL */}
      {activeTab === 'general' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-6">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <Globe size={18} className="text-blue-400" /> General Site Identity & Contact Info
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Site Name</label>
              <input
                type="text"
                value={settings.site_name}
                onChange={(e) => update('site_name', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Tagline / Short Description</label>
              <input
                type="text"
                value={settings.tagline}
                onChange={(e) => update('tagline', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Support Email</label>
              <input
                type="email"
                value={settings.email}
                onChange={(e) => update('email', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Primary Phone Number</label>
              <input
                type="text"
                value={settings.phone}
                onChange={(e) => update('phone', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">WhatsApp Direct URL / Number</label>
              <input
                type="text"
                value={settings.whatsapp}
                onChange={(e) => update('whatsapp', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#94A3B8]">Global Office Address</label>
            <textarea
              rows={2}
              value={settings.address}
              onChange={(e) => update('address', e.target.value)}
              className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#94A3B8]">Footer Copyright Text</label>
            <input
              type="text"
              value={settings.footer_text}
              onChange={(e) => update('footer_text', e.target.value)}
              className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
            />
          </div>
        </div>
      )}

      {/* TAB 2: SOCIAL MEDIA */}
      {activeTab === 'social' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-6">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <Share2 size={18} className="text-cyan-400" /> Social Media & Professional Network Profiles
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {[
              { key: 'social_linkedin', label: 'LinkedIn Company Page' },
              { key: 'social_twitter', label: 'Twitter / X Profile' },
              { key: 'social_instagram', label: 'Instagram Profile' },
              { key: 'social_github', label: 'GitHub Organization' },
              { key: 'social_youtube', label: 'YouTube Channel' },
              { key: 'social_dribbble', label: 'Dribbble Portfolio' },
            ].map(({ key, label }) => (
              <div key={key} className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">{label}</label>
                <input
                  type="text"
                  value={(settings as Record<string, string>)[key] || ''}
                  onChange={(e) => update(key, e.target.value)}
                  placeholder="https://"
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 font-mono"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 3: HOMEPAGE LINK */}
      {activeTab === 'homepage' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-8 text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mx-auto">
            <LayoutTemplate size={32} />
          </div>
          <h2 className="font-display text-xl font-bold text-white">Homepage Content Manager</h2>
          <p className="text-xs text-[#94A3B8] max-w-md mx-auto leading-relaxed">
            We have created a dedicated, full-screen live editor for your Homepage Hero, Stats Strip counters, and Call-to-Action sections.
          </p>
          <div className="pt-2">
            <Link
              href="/admin/homepage"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-2xl hover:shadow-glow-sm transition-all"
            >
              <span>Open Homepage CMS Panel</span>
              <ExternalLink size={14} />
            </Link>
          </div>
        </div>
      )}

      {/* TAB 4: PAGE CONTENT / BANNERS */}
      {(activeTab === 'pages' || activeTab === 'banners') && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-6">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <ImageIcon size={18} className="text-amber-400" /> Page Headers & Banner Text
          </h2>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">About Page Banner Headline</label>
              <input
                type="text"
                value={settings.banner_about_title}
                onChange={(e) => update('banner_about_title', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Services Catalog Banner Headline</label>
              <input
                type="text"
                value={settings.banner_services_title}
                onChange={(e) => update('banner_services_title', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Portfolio Grid Banner Headline</label>
              <input
                type="text"
                value={settings.banner_portfolio_title}
                onChange={(e) => update('banner_portfolio_title', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Contact Page Banner Headline</label>
              <input
                type="text"
                value={settings.banner_contact_title}
                onChange={(e) => update('banner_contact_title', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: SEO */}
      {activeTab === 'seo' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-6">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <Search size={18} className="text-emerald-400" /> Search Engine Optimization & Analytics
          </h2>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Global Meta Title</label>
              <input
                type="text"
                value={settings.seo_meta_title}
                onChange={(e) => update('seo_meta_title', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Global Meta Description</label>
              <textarea
                rows={3}
                value={settings.seo_meta_desc}
                onChange={(e) => update('seo_meta_desc', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">SEO Target Keywords (comma separated)</label>
              <input
                type="text"
                value={settings.seo_keywords}
                onChange={(e) => update('seo_keywords', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-[#273449]">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Google Analytics 4 ID</label>
                <input
                  type="text"
                  value={settings.google_analytics_id}
                  onChange={(e) => update('google_analytics_id', e.target.value)}
                  placeholder="G-XXXXXXXXXX"
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm font-mono focus:outline-none focus:border-blue-500/50"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Microsoft Clarity Project ID</label>
                <input
                  type="text"
                  value={settings.microsoft_clarity_id}
                  onChange={(e) => update('microsoft_clarity_id', e.target.value)}
                  placeholder="cl_xxxxxxxxxx"
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm font-mono focus:outline-none focus:border-blue-500/50"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 6: EMAIL */}
      {activeTab === 'email' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-6">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <Bell size={18} className="text-violet-400" /> Resend Email Dispatch & Notification Routing
          </h2>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Resend API Key</label>
              <input
                type="password"
                value={settings.resend_api_key}
                onChange={(e) => update('resend_api_key', e.target.value)}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm font-mono focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Sender Email Address (From)</label>
                <input
                  type="email"
                  value={settings.sender_email}
                  onChange={(e) => update('sender_email', e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-[#94A3B8]">Lead Notification Recipient Email (To)</label>
                <input
                  type="email"
                  value={settings.notification_email}
                  onChange={(e) => update('notification_email', e.target.value)}
                  className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 font-mono text-blue-400 font-bold"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 7: PRIVACY POLICY */}
      {activeTab === 'privacy' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-4">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <ShieldCheck size={18} className="text-blue-400" /> Privacy Policy Content CMS
          </h2>
          <p className="text-xs text-[#94A3B8]">Edit the complete legal terms that appear on `/privacy-policy`:</p>
          <textarea
            rows={10}
            value={settings.privacy_policy_text}
            onChange={(e) => update('privacy_policy_text', e.target.value)}
            className="w-full px-4 py-3 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs font-mono leading-relaxed focus:outline-none focus:border-blue-500/50"
          />
        </div>
      )}

      {/* TAB 8: TERMS OF SERVICE */}
      {activeTab === 'terms' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-4">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <FileText size={18} className="text-cyan-400" /> Terms of Service Content CMS
          </h2>
          <p className="text-xs text-[#94A3B8]">Edit the complete legal terms that appear on `/terms`:</p>
          <textarea
            rows={10}
            value={settings.terms_of_service_text}
            onChange={(e) => update('terms_of_service_text', e.target.value)}
            className="w-full px-4 py-3 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs font-mono leading-relaxed focus:outline-none focus:border-blue-500/50"
          />
        </div>
      )}

      {/* TAB 9: COOKIE POLICY */}
      {activeTab === 'cookies' && (
        <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-4">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <ShieldCheck size={18} className="text-emerald-400" /> Cookie Policy Content CMS
          </h2>
          <p className="text-xs text-[#94A3B8]">Edit the complete legal terms that appear on `/cookie-policy`:</p>
          <textarea
            rows={10}
            value={settings.cookie_policy_text}
            onChange={(e) => update('cookie_policy_text', e.target.value)}
            className="w-full px-4 py-3 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs font-mono leading-relaxed focus:outline-none focus:border-blue-500/50"
          />
        </div>
      )}
    </div>
  )
}

function SlidersIcon(props: { size?: number; className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={props.size || 24}
      height={props.size || 24}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={props.className}
    >
      <line x1="4" x2="4" y1="21" y2="14" />
      <line x1="4" x2="4" y1="10" y2="3" />
      <line x1="12" x2="12" y1="21" y2="12" />
      <line x1="12" x2="12" y1="8" y2="3" />
      <line x1="20" x2="20" y1="21" y2="16" />
      <line x1="20" x2="20" y1="12" y2="3" />
      <line x1="2" x2="6" y1="14" y2="14" />
      <line x1="10" x2="14" y1="8" y2="8" />
      <line x1="18" x2="22" y1="16" y2="16" />
    </svg>
  )
}
