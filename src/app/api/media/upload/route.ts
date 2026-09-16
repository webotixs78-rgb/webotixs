import { NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File | null
    const folder = (formData.get('folder') as string) || 'uploads'

    if (!file) {
      return NextResponse.json({ success: false, error: 'No file provided' }, { status: 400 })
    }

    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)
    const fileName = `${Date.now()}-${file.name.replace(/\s+/g, '-')}`
    const filePath = `${folder}/${fileName}`

    // Attempt upload to Supabase Storage using Admin Client (service role bypasses RLS)
    try {
      const supabase = createAdminClient()

      const { error: uploadError } = await supabase.storage
        .from('media')
        .upload(filePath, buffer, {
          contentType: file.type || 'application/octet-stream',
          upsert: true,
        })

      if (uploadError) {
        console.error('[Supabase Storage Upload Error]:', uploadError)
        return NextResponse.json({ success: false, error: uploadError.message || 'Upload failed' }, { status: 500 })
      }

      const { data } = supabase.storage.from('media').getPublicUrl(filePath)
      if (data?.publicUrl) {
        return NextResponse.json({ success: true, url: data.publicUrl, source: 'supabase_storage' })
      }
    } catch (e: any) {
      console.error('[Supabase Storage Upload Exception]:', e)
      return NextResponse.json({ success: false, error: e?.message || 'Server upload failed' }, { status: 500 })
    }

    return NextResponse.json({ success: false, error: 'Could not generate public URL' }, { status: 500 })
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message || 'Server error' }, { status: 500 })
  }
}
