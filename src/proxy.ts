import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function proxy(request: NextRequest) {
  let response = NextResponse.next({
    request: {
      headers: request.headers,
    },
  })

  // Check if predefined/demo admin session cookie is set
  const hasDemoSession = request.cookies.get('webotixs_admin_session')?.value === 'true'

  let user = null
  try {
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co',
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'placeholder',
      {
        cookies: {
          getAll() {
            return request.cookies.getAll()
          },
          setAll(cookiesToSet) {
            cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value))
            response = NextResponse.next({
              request,
            })
            cookiesToSet.forEach(({ name, value, options }) =>
              response.cookies.set(name, value, options)
            )
          },
        },
      }
    )

    const {
      data: { user: supabaseUser },
    } = await supabase.auth.getUser()
    user = supabaseUser
  } catch (err) {
    // If Supabase URL is placeholder or network fails, ignore and rely on demo session
  }

  const isAuthenticated = Boolean(user || hasDemoSession)

  // Protect admin routes
  if (request.nextUrl.pathname.startsWith('/admin')) {
    // Exclude the login page itself to prevent redirect loops
    if (request.nextUrl.pathname === '/admin/login') {
      if (isAuthenticated) {
        return NextResponse.redirect(new URL('/admin/dashboard', request.url))
      }
      return response
    }

    if (!isAuthenticated) {
      return NextResponse.redirect(new URL('/admin/login', request.url))
    }
  }

  return response
}

export const config = {
  matcher: ['/admin/:path*'],
}
