// Schema.org JSON-LD Structured Data for Webotixs
// Source: webotixs-service-schema.html (blocks 0-7)
// Note: "logo" omitted – no real logo image file exists in /public

export const organizationSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': ['Organization', 'ProfessionalService'],
  '@id': 'https://www.webotixs.com/#organization',
  name: 'Webotixs',
  url: 'https://www.webotixs.com/',
  image: 'https://webotixs.com/og-image.jpg',
  description:
    'Webotixs is a premium web design and digital agency crafting websites, mobile apps, and brand identities that drive real business results.',
  email: 'info@webotixs.com',
  telephone: '+12089055973',
  contactPoint: [
    {
      '@type': 'ContactPoint',
      telephone: '+12089055973',
      contactType: 'sales',
      email: 'info@webotixs.com',
      availableLanguage: 'English',
    },
    {
      '@type': 'ContactPoint',
      telephone: '+923092715559',
      contactType: 'customer support',
      availableLanguage: ['English', 'Urdu'],
    },
  ],
  address: {
    '@type': 'PostalAddress',
    streetAddress: 'Mz floor, Al-Qadir Heights, Kalma Chowk Flyover, Babar Block, Garden Town',
    addressLocality: 'Lahore',
    postalCode: '54000',
    addressCountry: 'PK',
  },
  sameAs: [
    'https://www.facebook.com/profile.php?id=61585230680319',
    'https://www.instagram.com/webotixs',
    'https://clutch.co/profile/webotixs',
    'https://www.trustpilot.com/review/webotixs.com',
    'https://g.page/r/Cbbz0-7qGKZVEBM/review',
  ],
}

// ── Block 1: Web Design & Development ────────────────────────────────────────

const webDesignSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'Service',
  '@id': 'https://www.webotixs.com/services/web-design-development#service',
  name: 'Web Design & Development',
  serviceType: 'Web Design and Development',
  url: 'https://www.webotixs.com/services/web-design-development',
  description:
    'Pixel-perfect, responsive, performance-optimized and SEO-ready websites that combine beautiful design with modern technology such as Next.js, React, and WordPress. Our process covers discovery, design, development, and launch.',
  image:
    'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/media/uploads/1784897003069-ChatGPT-Image-Jul-24,-2026,-05_03_20-PM.png',
  provider: { '@id': 'https://www.webotixs.com/#organization' },
  areaServed: [
    { '@type': 'Country', name: 'United States' },
    { '@type': 'Country', name: 'United Kingdom' },
    { '@type': 'Country', name: 'Canada' },
    { '@type': 'Country', name: 'Australia' },
  ],
  audience: { '@type': 'BusinessAudience', audienceType: 'Startups, SMEs and enterprises' },
  hasOfferCatalog: {
    '@type': 'OfferCatalog',
    name: 'Web Design & Development Services',
    itemListElement: [
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Responsive Website Design' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Performance-Optimized Development' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'SEO-Ready Website Build' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Conversion-Focused Web Design' } },
    ],
  },
  termsOfService: 'https://www.webotixs.com/terms',
  potentialAction: {
    '@type': 'ContactAction',
    target: 'https://www.webotixs.com/contact',
  },
}

// ── Block 2: Mobile App Development ──────────────────────────────────────────

const mobileAppSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'Service',
  '@id': 'https://www.webotixs.com/services/mobile-app-development#service',
  name: 'Mobile App Development',
  serviceType: 'Mobile Application Development',
  url: 'https://www.webotixs.com/services/mobile-app-development',
  description:
    'Native and cross-platform mobile apps for iOS and Android built with React Native and Flutter, including App Store Optimization to help users discover your app.',
  image:
    'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/media/uploads/1784898717646-ChatGPT-Image-Jul-24,-2026,-05_20_28-PM.png',
  provider: { '@id': 'https://www.webotixs.com/#organization' },
  areaServed: [
    { '@type': 'Country', name: 'United States' },
    { '@type': 'Country', name: 'United Kingdom' },
    { '@type': 'Country', name: 'Canada' },
    { '@type': 'Country', name: 'Australia' },
  ],
  audience: { '@type': 'BusinessAudience', audienceType: 'Startups, SMEs and enterprises' },
  hasOfferCatalog: {
    '@type': 'OfferCatalog',
    name: 'Mobile App Development Services',
    itemListElement: [
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'iOS & Android App Development' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'React Native App Development' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Flutter App Development' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'App Store Optimization' } },
    ],
  },
  termsOfService: 'https://www.webotixs.com/terms',
  potentialAction: {
    '@type': 'ContactAction',
    target: 'https://www.webotixs.com/contact',
  },
}

// ── Block 3: Brand Identity & Design ─────────────────────────────────────────

const brandIdentitySchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'Service',
  '@id': 'https://www.webotixs.com/services/brand-identity-design#service',
  name: 'Brand Identity & Design',
  serviceType: 'Brand Identity Design',
  url: 'https://www.webotixs.com/services/brand-identity-design',
  description:
    'Comprehensive brand identities that communicate your values and connect with your audience, including logo design, brand guidelines, visual identity systems, and brand strategy.',
  image:
    'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/media/uploads/1784897635729-ChatGPT-Image-Jul-24,-2026,-05_55_38-PM.png',
  provider: { '@id': 'https://www.webotixs.com/#organization' },
  areaServed: [
    { '@type': 'Country', name: 'United States' },
    { '@type': 'Country', name: 'United Kingdom' },
    { '@type': 'Country', name: 'Canada' },
    { '@type': 'Country', name: 'Australia' },
  ],
  audience: { '@type': 'BusinessAudience', audienceType: 'Startups and growing businesses' },
  hasOfferCatalog: {
    '@type': 'OfferCatalog',
    name: 'Brand Identity & Design Services',
    itemListElement: [
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Logo Design' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Brand Guidelines' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Visual Identity System' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Brand Strategy' } },
    ],
  },
  termsOfService: 'https://www.webotixs.com/terms',
  potentialAction: {
    '@type': 'ContactAction',
    target: 'https://www.webotixs.com/contact',
  },
}

// ── Block 4: E-Commerce Solutions ────────────────────────────────────────────

const ecommerceSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'Service',
  '@id': 'https://www.webotixs.com/services/ecommerce-solutions#service',
  name: 'E-Commerce Solutions',
  serviceType: 'E-Commerce Development',
  url: 'https://www.webotixs.com/services/ecommerce-solutions',
  description:
    'Powerful online stores built on Shopify, WooCommerce, or fully custom platforms with seamless payment integration, real-time inventory management, and conversion-rate optimization.',
  image:
    'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/media/uploads/1784897779842-ChatGPT-Image-Jul-24,-2026,-05_57_37-PM.png',
  provider: { '@id': 'https://www.webotixs.com/#organization' },
  areaServed: [
    { '@type': 'Country', name: 'United States' },
    { '@type': 'Country', name: 'United Kingdom' },
    { '@type': 'Country', name: 'Canada' },
    { '@type': 'Country', name: 'Australia' },
  ],
  audience: { '@type': 'BusinessAudience', audienceType: 'Retailers and e-commerce brands' },
  hasOfferCatalog: {
    '@type': 'OfferCatalog',
    name: 'E-Commerce Solutions',
    itemListElement: [
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Shopify Store Development' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'WooCommerce Development' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Custom E-Commerce Platform' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Payment Gateway Integration' } },
    ],
  },
  termsOfService: 'https://www.webotixs.com/terms',
  potentialAction: {
    '@type': 'ContactAction',
    target: 'https://www.webotixs.com/contact',
  },
}

// ── Block 5: SEO & Digital Marketing ─────────────────────────────────────────

const seoMarketingSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'Service',
  '@id': 'https://www.webotixs.com/services/seo-digital-marketing#service',
  name: 'SEO & Digital Marketing',
  serviceType: 'SEO and Digital Marketing',
  url: 'https://www.webotixs.com/services/seo-digital-marketing',
  description:
    'Data-driven SEO and digital marketing including technical and local SEO, content marketing, PPC and paid advertising, and analytics and growth tracking to increase visibility and qualified leads.',
  image:
    'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/media/uploads/1784897981685-ChatGPT-Image-Jul-24,-2026,-05_59_00-PM.png',
  provider: { '@id': 'https://www.webotixs.com/#organization' },
  areaServed: [
    { '@type': 'Country', name: 'United States' },
    { '@type': 'Country', name: 'United Kingdom' },
    { '@type': 'Country', name: 'Canada' },
    { '@type': 'Country', name: 'Australia' },
  ],
  audience: { '@type': 'BusinessAudience', audienceType: 'Local businesses, e-commerce brands and enterprises' },
  hasOfferCatalog: {
    '@type': 'OfferCatalog',
    name: 'SEO & Digital Marketing Services',
    itemListElement: [
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Technical & Local SEO Optimization' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Content Marketing' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'PPC & Paid Advertising Campaigns' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Analytics & Growth Tracking' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Social Media Marketing' } },
    ],
  },
  termsOfService: 'https://www.webotixs.com/terms',
  potentialAction: {
    '@type': 'ContactAction',
    target: 'https://www.webotixs.com/contact',
  },
}

// ── Block 6: Cloud & DevOps ───────────────────────────────────────────────────

const cloudDevopsSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'Service',
  '@id': 'https://www.webotixs.com/services/cloud-devops#service',
  name: 'Cloud & DevOps Solutions',
  serviceType: 'Cloud Architecture and DevOps',
  url: 'https://www.webotixs.com/services/cloud-devops',
  description:
    'Scalable, secure cloud infrastructure on AWS, Google Cloud, and Microsoft Azure with automated CI/CD pipelines, Docker and Kubernetes containerization, and 24/7 server monitoring and security audits.',
  image:
    'https://cicuwrgeqcnbyukkbapy.supabase.co/storage/v1/object/public/media/uploads/1784898441221-ChatGPT-Image-Jul-24,-2026,-06_07_06-PM.png',
  provider: { '@id': 'https://www.webotixs.com/#organization' },
  areaServed: [
    { '@type': 'Country', name: 'United States' },
    { '@type': 'Country', name: 'United Kingdom' },
    { '@type': 'Country', name: 'Canada' },
    { '@type': 'Country', name: 'Australia' },
  ],
  audience: { '@type': 'BusinessAudience', audienceType: 'SaaS companies and enterprises' },
  hasOfferCatalog: {
    '@type': 'OfferCatalog',
    name: 'Cloud & DevOps Services',
    itemListElement: [
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Multi-Cloud Architecture (AWS, GCP, Azure)' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Automated CI/CD Deployment Pipelines' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: 'Docker & Kubernetes Containerization' } },
      { '@type': 'Offer', itemOffered: { '@type': 'Service', name: '24/7 Server Monitoring & Security Audits' } },
    ],
  },
  termsOfService: 'https://www.webotixs.com/terms',
  potentialAction: {
    '@type': 'ContactAction',
    target: 'https://www.webotixs.com/contact',
  },
}

// ── Block 7: /services CollectionPage ────────────────────────────────────────

export const servicesHubSchema: Record<string, unknown> = {
  '@context': 'https://schema.org',
  '@type': 'CollectionPage',
  '@id': 'https://www.webotixs.com/services#page',
  url: 'https://www.webotixs.com/services',
  name: 'Enterprise Digital Services & Engineering | Webotixs',
  isPartOf: { '@id': 'https://www.webotixs.com/#organization' },
  mainEntity: {
    '@type': 'ItemList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, item: { '@id': 'https://www.webotixs.com/services/web-design-development#service' } },
      { '@type': 'ListItem', position: 2, item: { '@id': 'https://www.webotixs.com/services/mobile-app-development#service' } },
      { '@type': 'ListItem', position: 3, item: { '@id': 'https://www.webotixs.com/services/brand-identity-design#service' } },
      { '@type': 'ListItem', position: 4, item: { '@id': 'https://www.webotixs.com/services/ecommerce-solutions#service' } },
      { '@type': 'ListItem', position: 5, item: { '@id': 'https://www.webotixs.com/services/seo-digital-marketing#service' } },
      { '@type': 'ListItem', position: 6, item: { '@id': 'https://www.webotixs.com/services/cloud-devops#service' } },
    ],
  },
}

// ── Slug → schema map (used by /services/[slug]) ──────────────────────────────

export const serviceSchemaMap: Record<string, Record<string, unknown>> = {
  'web-design-development': webDesignSchema,
  'mobile-app-development': mobileAppSchema,
  'brand-identity-design': brandIdentitySchema,
  'ecommerce-solutions': ecommerceSchema,
  'seo-digital-marketing': seoMarketingSchema,
  'cloud-devops': cloudDevopsSchema,
}
