import { Metadata } from 'next'
import LocationLandingPage from '@/components/public/locations/LocationLandingPage'
import { locationData } from '@/lib/data/locationData'

const config = locationData['leeds-uk']

export const metadata: Metadata = {
  title: {
    absolute: config.pageTitle,
  },
  description: config.metaDescription,
  alternates: {
    canonical: config.canonicalUrl,
  },
  openGraph: {
    title: config.pageTitle,
    description: config.metaDescription,
    url: config.canonicalUrl,
    siteName: 'Webotixs',
    images: [
      {
        url: 'https://www.webotixs.com/og-image.jpg',
        width: 1200,
        height: 630,
        alt: config.pageTitle,
      },
    ],
    locale: 'en_GB',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: config.pageTitle,
    description: config.metaDescription,
    images: ['https://www.webotixs.com/og-image.jpg'],
  },
}

export default function LeedsLocationPage() {
  return <LocationLandingPage config={config} />
}
