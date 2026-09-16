import Header from '@/components/public/layout/Header'
import Footer from '@/components/public/layout/Footer'
import CustomCursor from '@/components/animations/CustomCursor'
import ParticleBackground from '@/components/animations/ParticleBackground'
import FixedFloatingWidget from '@/components/public/layout/FixedFloatingWidget'

export default function PublicLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <CustomCursor />
      <ParticleBackground />
      <Header />
      <FixedFloatingWidget />
      <main className="relative z-10">
        {children}
      </main>
      <Footer />
    </>
  )
}
