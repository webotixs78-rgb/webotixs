import HeroSection from '@/components/public/home/HeroSection'
import WhyChooseUsSection from '@/components/public/home/WhyChooseUsSection'
import ImpactNumbersSection from '@/components/public/home/ImpactNumbersSection'
import ServicesSection from '@/components/public/home/ServicesSection'
import PortfolioSection from '@/components/public/home/PortfolioSection'
import TestimonialsSection from '@/components/public/home/TestimonialsSection'
import TeamPreviewSection from '@/components/public/home/TeamPreviewSection'
import ContactSection from '@/components/public/home/ContactSection'

export default function Home() {
  return (
    <>
      <HeroSection />
      <WhyChooseUsSection />
      <ImpactNumbersSection />
      <ServicesSection />
      <PortfolioSection />
      <TestimonialsSection />
      <TeamPreviewSection />
      <ContactSection />
    </>
  )
}
