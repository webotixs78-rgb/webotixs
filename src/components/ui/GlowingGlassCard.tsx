'use client'

import React, { useRef } from 'react'
import { cn } from '@/lib/utils'

interface GlowingGlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
  className?: string
  glowColor?: string
}

export default function GlowingGlassCard({
  children,
  className,
  glowColor,
  onMouseMove,
  ...props
}: GlowingGlassCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      cardRef.current.style.setProperty('--mouse-x', `${x}px`)
      cardRef.current.style.setProperty('--mouse-y', `${y}px`)
    }
    if (onMouseMove) {
      onMouseMove(e)
    }
  }

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className={cn('glowing-glass-card group relative', className)}
      {...props}
    >
      <div className="relative z-10 w-full h-full">{children}</div>
    </div>
  )
}
