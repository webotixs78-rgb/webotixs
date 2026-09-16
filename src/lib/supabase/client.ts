import { createBrowserClient } from '@supabase/ssr'

export function createClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://cicuwrgeqcnbyukkbapy.supabase.co'
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'sb_publishable_KAcqG_MXX6pGSs3RWkVrjg_ZpbhZ6gg'

  return createBrowserClient(url, key)
}
