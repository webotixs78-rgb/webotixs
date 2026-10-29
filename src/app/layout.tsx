import { Space_Grotesk, Inter } from 'next/font/google'
import type { Metadata } from 'next'
import './globals.css'
import { ThemeProvider } from '@/components/providers/ThemeProvider'

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap',
})

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

export const metadata: Metadata = {
  title: {
    default: 'Webotixs — Premium Web Design & Development Agency',
    template: '%s | Webotixs',
  },
  description:
    'Crafting stunning digital experiences that convert visitors into customers. Websites, AI automation, and custom CRM solutions for growing businesses.',
  keywords: [
    'web design agency',
    'web development',
    'mobile app development',
    'brand identity',
    'AI automation',
    'custom CRM solutions',
    'digital agency',
    'SEO',
    'e-commerce',
    'Next.js',
    'Webotixs',
  ],
  authors: [{ name: 'Webotixs' }],
  creator: 'Webotixs',
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://www.webotixs.com'),
  icons: {
    icon: '/icon?v=2',
    shortcut: '/icon?v=2',
    apple: '/apple-icon?v=2',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://www.webotixs.com/',
    siteName: 'Webotixs',
    title: 'Webotixs | Premium Web Design & Development Agency',
    description:
      'We build high-performing websites, AI automations, and custom CRM solutions that help businesses grow.',
    images: [
      {
        url: 'https://www.webotixs.com/og-webotixs.jpg?v=2',
        secureUrl: 'https://www.webotixs.com/og-webotixs.jpg?v=2',
        type: 'image/jpeg',
        width: 1200,
        height: 630,
        alt: 'Webotixs — Premium Web Design, AI Automation & CRM Solutions',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Webotixs | Premium Web Design & Development Agency',
    description:
      'We build high-performing websites, AI automations, and custom CRM solutions that help businesses grow.',
    images: [
      {
        url: 'https://www.webotixs.com/og-webotixs.jpg?v=2',
        secureUrl: 'https://www.webotixs.com/og-webotixs.jpg?v=2',
        type: 'image/jpeg',
        width: 1200,
        height: 630,
        alt: 'Webotixs premium digital agency social preview.',
      },
    ],
    creator: '@webotixs',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${inter.variable} dark`}
      suppressHydrationWarning
    >
      <body className="font-body bg-background text-text-white antialiased" suppressHydrationWarning>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
