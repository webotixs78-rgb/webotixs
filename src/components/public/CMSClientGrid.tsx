'use client'

import { useState, useEffect } from 'react'
import { getCMSData } from '@/lib/data/cms'

interface CMSClientGridProps<T = any> {
  section: string
  initialData: T[]
  renderItem: (item: T, index: number) => React.ReactNode
  className?: string
  limit?: number
}

export default function CMSClientGrid<T = any>({
  section,
  initialData,
  renderItem,
  className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8',
  limit,
}: CMSClientGridProps<T>) {
  const [items, setItems] = useState<T[]>(() => {
    if (limit && Array.isArray(initialData)) return initialData.slice(0, limit)
    return initialData || []
  })

  useEffect(() => {
    let isMounted = true

    const loadData = () => {
      getCMSData<T[]>(section).then((data) => {
        if (!isMounted) return
        if (Array.isArray(data) && data.length > 0) {
          setItems(limit ? data.slice(0, limit) : data)
        }
      })
    }

    loadData()

    const handleStorage = () => {
      try {
        const local = localStorage.getItem(`webotixs_cms_${section}`)
        if (local) {
          const parsed = JSON.parse(local)
          if (Array.isArray(parsed) && parsed.length > 0) {
            setItems(limit ? parsed.slice(0, limit) : parsed)
          }
        }
      } catch {}
    }

    window.addEventListener('storage', handleStorage)
    return () => {
      isMounted = false
      window.removeEventListener('storage', handleStorage)
    }
  }, [section, limit])

  if (!items || items.length === 0) {
    return null
  }

  return (
    <div className={className}>
      {items.map((item, index) => renderItem(item, index))}
    </div>
  )
}
