import ScrollReveal from '@/components/animations/ScrollReveal'

export const metadata = {
  title: 'Terms of Service',
  description: 'Webotixs terms of service. Read the terms regulating the use of our services and website.',
}

export default function TermsPage() {
  return (
    <div className="pt-24 bg-background min-h-screen">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <ScrollReveal className="mb-10">
          <h1 className="font-display text-4xl font-bold text-text-white mb-4">Terms of Service</h1>
          <p className="text-text-gray text-xs">Last Updated: July 16, 2026</p>
        </ScrollReveal>

        <ScrollReveal className="prose prose-invert text-text-gray space-y-6 text-sm leading-relaxed">
          <p>
            Welcome to Webotixs! These terms and conditions outline the rules and regulations for the use of Webotixs&apos;s Website, located at webotixs.com.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">1. Agreement to Terms</h2>
          <p>
            By accessing this website we assume you accept these terms and conditions. Do not continue to use Webotixs if you do not agree to take all of the terms and conditions stated on this page.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">2. Intellectual Property Rights</h2>
          <p>
            Other than the content you own, under these Terms, Webotixs and/or its licensors own all the intellectual property rights and materials contained in this Website. You are granted limited license only for purposes of viewing the material contained on this Website.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">3. Restrictions</h2>
          <p>You are specifically restricted from all of the following:</p>
          <ul className="list-disc pl-6 space-y-1.5">
            <li>Publishing any Website material in any other media</li>
            <li>Selling, sublicensing and/or otherwise commercializing any Website material</li>
            <li>Publicly performing and/or showing any Website material</li>
            <li>Using this Website in any way that is or may be damaging to this Website</li>
            <li>Using this Website contrary to applicable laws and regulations</li>
          </ul>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">4. Limitation of Liability</h2>
          <p>
            In no event shall Webotixs, nor any of its officers, directors and employees, be held liable for anything arising out of or in any way connected with your use of this Website whether such liability is under contract.
          </p>
        </ScrollReveal>
      </div>
    </div>
  )
}
