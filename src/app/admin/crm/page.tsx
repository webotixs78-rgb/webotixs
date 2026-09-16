'use client'

import React, { Suspense } from 'react'
import { AgencyCRMClientHub } from '@/components/admin/crm/AgencyCRMClientHub'
import { Loader2 } from 'lucide-react'

export const dynamic = 'force-dynamic'

export default function AdminCRMPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh] text-[#94A3B8]">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500 mr-3" />
          <span>Loading Agency CRM Operating System...</span>
        </div>
      }
    >
      <AgencyCRMClientHub />
    </Suspense>
  )
}
