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
    'Webotixs is a premium web design and digital agency crafting stunning websites, mobile apps, and brand identities that drive real business results.',
  keywords: [
    'web design agency',
    'web development',
    'mobile app development',
    'brand identity',
    'digital agency',
    'SEO',
    'e-commerce',
    'Next.js',
    'Webotixs',
  ],
  authors: [{ name: 'Webotixs' }],
  creator: 'Webotixs',
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://webotixs.com'),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: process.env.NEXT_PUBLIC_APP_URL || 'https://webotixs.com',
    siteName: 'Webotixs',
    title: 'Webotixs — Premium Web Design & Development Agency',
    description:
      'Crafting stunning digital experiences that convert visitors into customers.',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'Webotixs Agency',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Webotixs — Premium Web Design & Development Agency',
    description:
      'Crafting stunning digital experiences that convert visitors into customers.',
    images: ['/og-image.jpg'],
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
