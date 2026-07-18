import Header from '@/components/public/layout/Header'
import Footer from '@/components/public/layout/Footer'
import CustomCursor from '@/components/animations/CustomCursor'
import ParticleBackground from '@/components/animations/ParticleBackground'

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
      <main className="relative z-10">
        {children}
      </main>
      <Footer />
    </>
  )
}
