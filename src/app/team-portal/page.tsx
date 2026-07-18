'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Loader2 } from 'lucide-react'

export default function TeamPortalRedirect() {
  const router = useRouter()
  useEffect(() => {
    router.replace('/team/dashboard')
  }, [router])

  return (
    <div className="min-h-screen bg-[#050816] flex items-center justify-center text-white p-6">
      <div className="flex items-center gap-3 text-xs text-purple-400">
        <Loader2 size={16} className="animate-spin" />
        <span>Redirecting to Staff Workspace Dashboard...</span>
      </div>
    </div>
  )
}
