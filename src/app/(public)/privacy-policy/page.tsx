import ScrollReveal from '@/components/animations/ScrollReveal'

export const metadata = {
  title: 'Privacy Policy',
  description: 'Webotixs privacy policy. Learn how we handle your personal data and respect privacy rules.',
}

export default function PrivacyPolicyPage() {
  return (
    <div className="pt-24 bg-background min-h-screen">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <ScrollReveal className="mb-10">
          <h1 className="font-display text-4xl font-bold text-text-white mb-4">Privacy Policy</h1>
          <p className="text-text-gray text-xs">Last Updated: July 16, 2026</p>
        </ScrollReveal>

        <ScrollReveal className="prose prose-invert text-text-gray space-y-6 text-sm leading-relaxed">
          <p>
            At Webotixs, accessible from webotixs.com, one of our main priorities is the privacy of our visitors. This Privacy Policy document contains types of information that is collected and recorded by Webotixs and how we use it.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">1. Information We Collect</h2>
          <p>
            The personal information that you are asked to provide, and the reasons why you are asked to provide it, will be made clear to you at the point we ask you to provide your personal information.
          </p>
          <p>
            If you contact us directly, we may receive additional information about you such as your name, email address, phone number, the contents of the message and/or attachments you may send us, and any other information you may choose to provide.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">2. How We Use Your Information</h2>
          <p>We use the information we collect in various ways, including to:</p>
          <ul className="list-disc pl-6 space-y-1.5">
            <li>Provide, operate, and maintain our website</li>
            <li>Improve, personalize, and expand our website</li>
            <li>Understand and analyze how you use our website</li>
            <li>Develop new products, services, features, and functionality</li>
            <li>Communicate with you for customer service, updates, and marketing</li>
          </ul>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">3. Log Files</h2>
          <p>
            Webotixs follows a standard procedure of using log files. These files log visitors when they visit websites. The information collected by log files include internet protocol (IP) addresses, browser type, Internet Service Provider (ISP), date and time stamp, referring/exit pages, and possibly the number of clicks.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">4. GDPR &amp; CCPA Privacy Rights</h2>
          <p>
            We want to make sure you are fully aware of all of your data protection rights. Every user is entitled to the rights to access, rectification, erasure, restrict processing, object to processing, and data portability.
          </p>
        </ScrollReveal>
      </div>
    </div>
  )
}
