'use client'

import { useEffect, useRef } from 'react'

export default function CustomCursor() {
  const dotRef = useRef<HTMLDivElement>(null)
  const outlineRef = useRef<HTMLDivElement>(null)
  const pos = useRef({ x: 0, y: 0 })
  const outlinePos = useRef({ x: 0, y: 0 })

  useEffect(() => {
    // Only show custom cursor on non-touch devices
    if (window.matchMedia('(pointer: coarse)').matches) return

    const dot = dotRef.current
    const outline = outlineRef.current
    if (!dot || !outline) return

    dot.style.display = 'block'
    outline.style.display = 'block'

    const onMove = (e: MouseEvent) => {
      pos.current = { x: e.clientX, y: e.clientY }
      dot.style.transform = `translate(${e.clientX - 4}px, ${e.clientY - 4}px)`
    }
    window.addEventListener('mousemove', onMove)

    let rafId: number
    const animate = () => {
      outlinePos.current.x += (pos.current.x - outlinePos.current.x) * 0.12
      outlinePos.current.y += (pos.current.y - outlinePos.current.y) * 0.12
      outline.style.transform = `translate(${outlinePos.current.x - 18}px, ${outlinePos.current.y - 18}px)`
      rafId = requestAnimationFrame(animate)
    }
    animate()

    const onEnter = () => outline.style.transform += ' scale(1.5)'
    const onLeave = () => outline.style.transform = outline.style.transform.replace(' scale(1.5)', '')

    document.querySelectorAll('a, button, [role="button"]').forEach((el) => {
      el.addEventListener('mouseenter', onEnter)
      el.addEventListener('mouseleave', onLeave)
    })

    return () => {
      window.removeEventListener('mousemove', onMove)
      cancelAnimationFrame(rafId)
    }
  }, [])

  return (
    <>
      <div
        ref={dotRef}
        className="cursor-dot"
        style={{ display: 'none' }}
      />
      <div
        ref={outlineRef}
        className="cursor-outline"
        style={{ display: 'none' }}
      />
    </>
  )
}
