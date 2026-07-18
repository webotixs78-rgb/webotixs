import Link from 'next/link'
import { ArrowLeft, Compass } from 'lucide-react'

export const metadata = {
  title: 'Page Not Found',
  description: 'The page you are looking for does not exist on the Webotixs platform.',
}

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      {/* Grid pattern overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.02]"
        style={{
          backgroundImage: `linear-gradient(rgba(248,250,252,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(248,250,252,0.3) 1px, transparent 1px)`,
          backgroundSize: '60px 60px',
        }}
      />

      <div className="relative text-center max-w-md mx-auto space-y-6 z-10">
        <div className="w-16 h-16 rounded-3xl bg-primary/10 flex items-center justify-center mx-auto text-primary border border-primary/20">
          <Compass size={28} className="animate-spin-slow" />
        </div>

        <h1 className="font-display text-7xl font-bold gradient-text">404</h1>
        <h2 className="font-display text-2xl font-bold text-text-white">Lost in Space?</h2>
        <p className="text-text-gray text-sm leading-relaxed">
          The page you are trying to visit has either been moved, deleted, or does not exist. Let&apos;s get you back on track.
        </p>

        <Link
          href="/"
          className="inline-flex items-center gap-2 px-6 py-3.5 bg-gradient-to-r from-primary-from to-primary-to text-white font-semibold rounded-2xl shadow-glow-sm hover:shadow-glow-md transition-all hover:scale-103 text-sm"
        >
          <ArrowLeft size={16} />
          Back to Homepage
        </Link>
      </div>
    </div>
  )
}
