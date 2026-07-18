'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Loader2 } from 'lucide-react'

export default function ClientPortalRedirect() {
  const router = useRouter()
  useEffect(() => {
    router.replace('/client/dashboard')
  }, [router])

  return (
    <div className="min-h-screen bg-[#050816] flex items-center justify-center text-white p-6">
      <div className="flex items-center gap-3 text-xs text-cyan-400">
        <Loader2 size={16} className="animate-spin" />
        <span>Redirecting to VIP Client Dashboard...</span>
      </div>
    </div>
  )
}
