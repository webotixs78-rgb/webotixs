'use client'

import React, { useState, useRef } from 'react'
import Link from 'next/link'
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Sparkles,
  Quote,
  Building,
  ArrowRight,
  Star,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react'
import GlowingGlassCard from '@/components/ui/GlowingGlassCard'

const SUPABASE_VIDEO_URL =
  'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/webotixs_cms_data/videos/felix-testimonial.mp4'

export default function VideoTestimonialSection() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [progress, setProgress] = useState(0)

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
        setIsPlaying(false)
      } else {
        const promise = videoRef.current.play()
        if (promise !== undefined) {
          promise
            .then(() => setIsPlaying(true))
            .catch((err) => {
              console.error('Playback error:', err)
              setIsPlaying(false)
            })
        }
      }
    }
  }

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted
      setIsMuted(!isMuted)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const current = videoRef.current.currentTime
      const duration = videoRef.current.duration || 1
      setProgress((current / duration) * 100)
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (videoRef.current) {
      const seekTime = (parseFloat(e.target.value) / 100) * (videoRef.current.duration || 1)
      videoRef.current.currentTime = seekTime
      setProgress(parseFloat(e.target.value))
    }
  }

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen()
      }
    }
  }

  return (
    <section className="py-24 relative overflow-hidden bg-background">
      {/* Ambient Glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-gradient-to-tr from-primary/15 to-cyan-500/15 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-mono font-bold uppercase tracking-wider">
            <Sparkles size={14} className="text-primary animate-pulse" />
            <span>Featured Client Video Story</span>
          </div>

          <h2 className="font-display text-3xl md:text-5xl font-bold text-text-white tracking-tight">
            See How We Helped <span className="gradient-text">Olmos Brothers Roofing</span> Scale
          </h2>

          <p className="text-text-gray text-base md:text-lg leading-relaxed">
            Listen directly to <strong className="text-white">Felix</strong>, Owner of <strong className="text-white">Olmos Brothers Roofing</strong>, share how Webotixs transformed their digital presence and scaled commercial leads.
          </p>
        </div>

        {/* Main Layout: Vertical Smartphone Device Frame + Client Details */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Vertical Smartphone Device Frame (9:16 Aspect Ratio) */}
          <div className="lg:col-span-5 flex justify-center">
            <GlowingGlassCard className="w-full max-w-[340px] rounded-[42px] p-3 border-2 border-border/80 bg-background-secondary/95 shadow-glow-md relative group">
              {/* Smartphone Frame Container */}
              <div className="relative rounded-[34px] overflow-hidden bg-black border border-white/10 aspect-[9/16] shadow-2xl flex flex-col justify-between">
                {/* Dynamic Island Notch */}
                <div className="absolute top-3 left-1/2 -translate-x-1/2 w-24 h-4 bg-black rounded-full z-30 flex items-center justify-center border border-white/10 pointer-events-none">
                  <div className="w-2.5 h-2.5 rounded-full bg-zinc-800 mr-2" />
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-900" />
                </div>

                {/* Local Video Player Element */}
                <div className="relative w-full h-full bg-black overflow-hidden flex items-center justify-center">
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    onTimeUpdate={handleTimeUpdate}
                    onEnded={() => setIsPlaying(false)}
                    playsInline
                    controls
                    preload="auto"
                  >
                    <source src={SUPABASE_VIDEO_URL} type="video/mp4" />
                    <source src="/videos/felix-testimonial.mp4" type="video/mp4" />
                    Your browser does not support HTML5 video streaming.
                  </video>

                  {/* Play Overlay (Shown when paused) */}
                  {!isPlaying && (
                    <div
                      onClick={togglePlay}
                      className="absolute inset-0 bg-black/40 backdrop-blur-[1px] flex flex-col items-center justify-center cursor-pointer transition-all hover:bg-black/20 group/play z-20"
                    >
                      <div className="w-16 h-16 rounded-full bg-gradient-to-r from-primary-from to-primary-to flex items-center justify-center text-white shadow-glow-lg group-hover/play:scale-110 transition-transform mb-3">
                        <Play size={28} className="ml-1 fill-current" />
                      </div>
                      <span className="text-xs font-bold text-white tracking-wide bg-black/70 px-4 py-1.5 rounded-full border border-white/20 backdrop-blur-md">
                        Click to Play Video
                      </span>
                    </div>
                  )}

                  {/* Mobile Custom Controls Bar */}
                  <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/90 via-black/60 to-transparent flex flex-col gap-1.5 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={progress}
                      onChange={handleSeek}
                      className="w-full h-1 bg-white/30 rounded-lg appearance-none cursor-pointer accent-primary"
                    />

                    <div className="flex items-center justify-between text-white text-xs pt-1">
                      <div className="flex items-center gap-3">
                        <button
                          onClick={togglePlay}
                          className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                          aria-label={isPlaying ? 'Pause' : 'Play'}
                        >
                          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                        </button>

                        <button
                          onClick={toggleMute}
                          className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                          aria-label={isMuted ? 'Unmute' : 'Mute'}
                        >
                          {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
                        </button>
                      </div>

                      <button
                        onClick={handleFullscreen}
                        className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                        aria-label="Fullscreen"
                      >
                        <Maximize size={14} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </GlowingGlassCard>
          </div>

          {/* Right Column: Client Case Study & Detailed Spotlight */}
          <div className="lg:col-span-7 space-y-6">
            <GlowingGlassCard className="p-8 rounded-3xl border border-border/80 bg-background-secondary/80 shadow-2xl relative">
              <Quote size={48} className="text-primary/20 absolute top-6 right-6 pointer-events-none" />

              {/* Company & Client Header */}
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-from to-primary-to flex items-center justify-center text-white shadow-lg text-xl font-bold font-display">
                  OB
                </div>
                <div>
                  <div className="font-display font-bold text-2xl text-text-white flex items-center gap-2">
                    Felix
                    <CheckCircle2 size={18} className="text-emerald-400" />
                  </div>
                  <p className="text-xs text-primary font-semibold flex items-center gap-1.5 mt-0.5">
                    <Building size={14} /> Owner, Olmos Brothers Roofing
                  </p>
                </div>
              </div>

              {/* Rating */}
              <div className="flex items-center gap-1 mb-4 text-amber-400">
                {[...Array(5)].map((_, i) => (
                  <Star key={i} size={16} className="fill-current" />
                ))}
                <span className="text-xs font-bold text-text-white ml-2">5.0 Outstanding Client Rating</span>
              </div>

              {/* Client Quote */}
              <blockquote className="text-text-gray text-base leading-relaxed italic mb-8 border-l-2 border-primary/50 pl-4 py-1">
                &ldquo;Webotixs built a high-converting digital platform that completely elevated our roofing business. The site looks incredible, loads instantly, and delivers steady commercial leads every month.&rdquo;
              </blockquote>

              {/* Key Highlights & Metrics */}
              <div className="grid grid-cols-2 gap-4 pt-6 border-t border-border/60 mb-8">
                <div className="bg-background/60 rounded-2xl p-4 border border-border/50">
                  <div className="text-2xl font-display font-bold gradient-text">+240%</div>
                  <div className="text-xs text-text-gray mt-0.5 font-medium">Commercial Lead Growth</div>
                </div>
                <div className="bg-background/60 rounded-2xl p-4 border border-border/50">
                  <div className="text-2xl font-display font-bold text-emerald-400">100%</div>
                  <div className="text-xs text-text-gray mt-0.5 font-medium">Client Satisfaction</div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row items-center gap-3">
                <Link
                  href="/contact"
                  className="btn-float-rtl w-full sm:w-auto flex-1 py-3.5 px-6 bg-gradient-to-r from-primary-from to-primary-to text-white font-bold text-xs rounded-xl shadow-glow-sm text-center flex items-center justify-center gap-2"
                >
                  Build Your Digital Platform <ArrowRight size={14} />
                </Link>
                <Link
                  href="/industries"
                  className="btn-float-rtl-glass w-full sm:w-auto py-3.5 px-5 glass border border-border text-text-white text-xs font-bold rounded-xl text-center flex items-center justify-center gap-1.5"
                >
                  View Industry Solutions <ExternalLink size={13} />
                </Link>
              </div>
            </GlowingGlassCard>
          </div>
        </div>
      </div>
    </section>
  )
}
