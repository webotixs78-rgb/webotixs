import ScrollReveal from '@/components/animations/ScrollReveal'

export const metadata = {
  title: 'Cookie Policy',
  description: 'Webotixs cookie policy. Learn how we use cookies and tracking techniques on our platform.',
}

export default function CookiePolicyPage() {
  return (
    <div className="pt-24 bg-background min-h-screen">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <ScrollReveal className="mb-10">
          <h1 className="font-display text-4xl font-bold text-text-white mb-4">Cookie Policy</h1>
          <p className="text-text-gray text-xs">Last Updated: July 16, 2026</p>
        </ScrollReveal>

        <ScrollReveal className="prose prose-invert text-text-gray space-y-6 text-sm leading-relaxed">
          <p>
            This is the Cookie Policy for Webotixs, accessible from webotixs.com. As is common practice with almost all professional websites this site uses cookies, which are tiny files that are downloaded to your computer, to improve your experience.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">1. How We Use Cookies</h2>
          <p>
            We use cookies for a variety of reasons detailed below. Unfortunately, in most cases, there are no industry standard options for disabling cookies without completely disabling the functionality and features they add to this site.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">2. Disabling Cookies</h2>
          <p>
            You can prevent the setting of cookies by adjusting the settings on your browser (see your browser Help for how to do this). Be aware that disabling cookies will affect the functionality of this and many other websites that you visit.
          </p>

          <h2 className="font-display text-lg font-bold text-text-white pt-4">3. The Cookies We Set</h2>
          <p>We set the following types of cookies:</p>
          <ul className="list-disc pl-6 space-y-1.5">
            <li><strong>Forms related cookies:</strong> When you submit data to through a form such as those found on contact pages or comment forms cookies may be set to remember your user details for future correspondence.</li>
            <li><strong>Site preferences cookies:</strong> In order to provide you with a great experience on this site we provide the functionality to set your preferences for how this site runs when you use it.</li>
          </ul>
        </ScrollReveal>
      </div>
    </div>
  )
}
