'use client'

import { useState, useEffect } from 'react'
import {
  Sparkles,
  Save,
  Loader2,
  CheckCircle2,
  LayoutTemplate,
  Type,
  TrendingUp,
  Sliders,
  Eye,
  RefreshCw,
} from 'lucide-react'
import { getCMSData, saveCMSList } from '@/lib/data/cms'
import { cn } from '@/lib/utils'

export default function AdminHomepageCMSPage() {
  const [saving, setSaving] = useState(false)
  const [successMsg, setSuccessMsg] = useState(false)
  const [activeTab, setActiveTab] = useState<'hero' | 'stats' | 'cta'>('hero')

  // Editable Homepage State
  const [hero, setHero] = useState<any>({
    badge: 'Trusted by 200+ Global Clients',
    badgeText: 'Trusted by 200+ Global Clients',
    titlePrefix: 'We Build',
    titleHighlight: 'Digital',
    titleSuffix: 'Experiences',
    titleMain: 'We Build Digital Experiences',
    subtitle:
      'Premium web design, mobile apps, and brand identities crafted for ambitious businesses. We turn your vision into stunning digital products.',
    primaryCtaText: 'Start Your Project',
    ctaPrimaryText: 'Start Your Project',
    primaryCtaLink: '/contact',
    secondaryCtaText: 'View Our Work',
    ctaSecondaryText: 'View Our Work',
    secondaryCtaLink: '/portfolio',
  })

  const [stats, setStats] = useState<any[]>([
    { id: '1', label: 'Successful Projects Launched', value: '150', suffix: '+' },
    { id: '2', label: 'Happy Clients Worldwide', value: '98', suffix: '%' },
    { id: '3', label: 'Years of Industry Experience', value: '10', suffix: '+' },
    { id: '4', label: 'Digital Impressions Generated', value: '25', suffix: 'M+' },
  ])

  const [ctaSection, setCtaSection] = useState<any>({
    badge: 'Get Started Today',
    heading: 'Ready to Scale Your Digital Business?',
    title: 'Ready to Scale Your Digital Business?',
    description:
      'Partner with Webotixs and build high-performance web applications tailored to your exact enterprise requirements.',
    subtitle:
      'Partner with Webotixs and build high-performance web applications tailored to your exact enterprise requirements.',
    buttonText: 'Schedule Consultation',
    buttonUrl: '/contact',
  })

  useEffect(() => {
    getCMSData('homepage').then((parsed) => {
      if (parsed?.hero) setHero(parsed.hero)
      if (parsed?.stats && Array.isArray(parsed.stats)) setStats(parsed.stats)
      if (parsed?.ctaSection) setCtaSection(parsed.ctaSection)
    })
  }, [])

  const handleSave = async () => {
    setSaving(true)
    await saveCMSList('homepage', { hero, stats, ctaSection })
    setSaving(false)
    setSuccessMsg(true)
    setTimeout(() => setSuccessMsg(false), 4000)
  }

  const handleResetDefaults = () => {
    if (!window.confirm('Reset Homepage CMS state to original production defaults?')) return
    localStorage.removeItem('webotixs_cms_homepage')
    fetch('/api/cms/homepage?id=main', { method: 'DELETE' }).then(() => {
      window.location.reload()
    })
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2.5">
            <Sparkles size={24} className="text-blue-500" /> Homepage CMS Manager
          </h1>
          <p className="text-[#94A3B8] text-xs mt-1">
            Real-time content management for the public homepage hero, stats strip, and call-to-action banners.
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
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold rounded-xl hover:shadow-glow-sm transition-all disabled:opacity-50"
          >
            {saving ? (
              <>
                <Loader2 size={14} className="animate-spin" /> Saving CMS...
              </>
            ) : (
              <>
                <Save size={14} /> Save Homepage Content
              </>
            )}
          </button>
        </div>
      </div>

      {successMsg && (
        <div className="px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold rounded-2xl flex items-center gap-2">
          <CheckCircle2 size={16} /> Homepage content updated successfully across live view!
        </div>
      )}

      {/* Hero Section Box */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-[#273449] pb-4">
          <h2 className="font-display text-base font-bold text-white flex items-center gap-2">
            <LayoutTemplate size={18} className="text-blue-400" /> Hero Section Settings
          </h2>
          <span className="text-[10px] text-emerald-400 font-mono bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
            Live Preview Active
          </span>
        </div>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#94A3B8]">Top Badge Tagline</label>
            <input
              type="text"
              value={hero.badgeText || hero.badge || ''}
              onChange={(e) => setHero({ ...hero, badgeText: e.target.value, badge: e.target.value })}
              className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Main Heading Text</label>
              <input
                type="text"
                value={hero.titleMain || hero.titlePrefix || ''}
                onChange={(e) => setHero({ ...hero, titleMain: e.target.value, titlePrefix: e.target.value })}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Gradient Highlight Words</label>
              <input
                type="text"
                value={hero.titleHighlight || ''}
                onChange={(e) => setHero({ ...hero, titleHighlight: e.target.value })}
                className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-blue-400 font-semibold text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#94A3B8]">Hero Description / Subtitle</label>
            <textarea
              rows={3}
              value={hero.subtitle || ''}
              onChange={(e) => setHero({ ...hero, subtitle: e.target.value })}
              className="w-full px-4 py-2.5 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm focus:outline-none focus:border-blue-500/50 resize-none"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Primary Button Text</label>
              <input
                type="text"
                value={hero.ctaPrimaryText || hero.primaryCtaText || ''}
                onChange={(e) => setHero({ ...hero, ctaPrimaryText: e.target.value, primaryCtaText: e.target.value })}
                className="w-full px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#94A3B8]">Secondary Button Text</label>
              <input
                type="text"
                value={hero.ctaSecondaryText || hero.secondaryCtaText || ''}
                onChange={(e) => setHero({ ...hero, ctaSecondaryText: e.target.value, secondaryCtaText: e.target.value })}
                className="w-full px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Stats Strip Box */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-5">
        <h2 className="font-display text-base font-bold text-white flex items-center gap-2 border-b border-[#273449] pb-4">
          <TrendingUp size={18} className="text-cyan-400" /> Stats Strip Counters
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {stats.map((item, idx) => (
            <div key={item.id} className="bg-[#050816] border border-[#273449] rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-blue-400">Counter #{idx + 1}</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="col-span-2 space-y-1">
                  <label className="text-[10px] text-[#94A3B8] font-bold uppercase">Number Value</label>
                  <input
                    type="text"
                    value={item.value}
                    onChange={(e) => {
                      const newStats = [...stats]
                      newStats[idx].value = e.target.value
                      setStats(newStats)
                    }}
                    className="w-full px-3 py-1.5 bg-[#0D1224] border border-[#273449] rounded-xl text-white text-sm font-bold"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-[#94A3B8] font-bold uppercase">Suffix</label>
                  <input
                    type="text"
                    value={item.suffix}
                    onChange={(e) => {
                      const newStats = [...stats]
                      newStats[idx].suffix = e.target.value
                      setStats(newStats)
                    }}
                    className="w-full px-3 py-1.5 bg-[#0D1224] border border-[#273449] rounded-xl text-cyan-400 text-sm font-bold text-center"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[10px] text-[#94A3B8] font-bold uppercase">Counter Label</label>
                <input
                  type="text"
                  value={item.label}
                  onChange={(e) => {
                    const newStats = [...stats]
                    newStats[idx].label = e.target.value
                    setStats(newStats)
                  }}
                  className="w-full px-3 py-1.5 bg-[#0D1224] border border-[#273449] rounded-xl text-xs text-white"
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom CTA Banner Box */}
      <div className="bg-[#0D1224] border border-[#273449] rounded-3xl p-6 space-y-4">
        <h2 className="font-display text-base font-bold text-white flex items-center gap-2 border-b border-[#273449] pb-4">
          <Type size={18} className="text-violet-400" /> Homepage Call-to-Action Banner
        </h2>

        <div className="space-y-3">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#94A3B8]">Banner Title</label>
            <input
              type="text"
              value={ctaSection.title || ctaSection.heading || ''}
              onChange={(e) => setCtaSection({ ...ctaSection, title: e.target.value, heading: e.target.value })}
              className="w-full px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-sm"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#94A3B8]">Banner Subtitle</label>
            <textarea
              rows={2}
              value={ctaSection.subtitle || ctaSection.description || ''}
              onChange={(e) => setCtaSection({ ...ctaSection, subtitle: e.target.value, description: e.target.value })}
              className="w-full px-4 py-2 bg-[#050816] border border-[#273449] rounded-xl text-white text-xs resize-none"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
