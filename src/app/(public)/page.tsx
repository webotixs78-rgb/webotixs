// v2 - glassmorphism portfolio & FAQ section
import HeroSection from '@/components/public/home/HeroSection'
import TechLogoSlider from '@/components/public/home/TechLogoSlider'
import WhyChooseUsSection from '@/components/public/home/WhyChooseUsSection'
import ImpactNumbersSection from '@/components/public/home/ImpactNumbersSection'
import ServicesSection from '@/components/public/home/ServicesSection'
import PortfolioSection from '@/components/public/home/PortfolioSection'
import FaqSection from '@/components/public/home/FaqSection'
import TestimonialsSection from '@/components/public/home/TestimonialsSection'
import VideoTestimonialSection from '@/components/public/home/VideoTestimonialSection'
import TeamPreviewSection from '@/components/public/home/TeamPreviewSection'
import ContactSection from '@/components/public/home/ContactSection'
import JsonLd from '@/components/JsonLd'
import { organizationSchema } from '@/lib/schema/service-schemas'

export const revalidate = 60

export default function Home() {
  return (
    <>
      <JsonLd data={organizationSchema} />
      <HeroSection />
      <TechLogoSlider />
      <WhyChooseUsSection />
      <ImpactNumbersSection />
      <ServicesSection />
      <PortfolioSection />
      <FaqSection />
      <TestimonialsSection />
      <VideoTestimonialSection />
      <TeamPreviewSection />
      <ContactSection />
    </>
  )
}
