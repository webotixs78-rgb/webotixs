export interface ServiceCardItem {
  title: string
  description: string
  icon: string
  href: string
}

export interface FAQItem {
  question: string
  answer: string
}

export interface LocationConfig {
  city: string
  state: string
  stateAbbr: string
  slug: string
  pageTitle: string
  metaDescription: string
  canonicalUrl: string
  eyebrow: string
  h1: string
  heroSupport: string
  locationReferenceBadge: string
  
  // Section 3: Introduction
  introHeading: string
  introParagraphs: string[]
  
  // Section 4: Services
  servicesHeading: string
  servicesSubheading: string
  services: ServiceCardItem[]
  
  // Section 5: Web Design & Development
  webDevHeading: string
  webDevCopy: string[]
  webDevFeatures: string[]
  
  // Section 6: SEO Section
  seoHeading: string
  seoCopy: string[]
  seoCapabilities: string[]
  
  // Section 7: Hosting & Maintenance
  hostingHeading: string
  hostingCopy: string[]
  hostingFeatures: string[]
  
  // Section 8: AI Chatbot & Automation
  aiHeading: string
  aiCopy: string[]
  aiUseCases: string[]
  
  // Section 9: CRM Services
  crmHeading: string
  crmCopy: string[]
  crmFeatures: string[]
  
  // Section 13: Local Relevance
  localAreaHeading: string
  localAreaIntro: string
  localAreas: string[]
  
  // Section 14: FAQs
  faqsHeading: string
  faqs: FAQItem[]
  
  // Section 15: Final CTA
  ctaHeading: string
  ctaSupport: string
}

export const locationData: Record<string, LocationConfig> = {
  'dallas-tx': {
    city: 'Dallas',
    state: 'Texas',
    stateAbbr: 'TX',
    slug: 'dallas-tx',
    pageTitle: 'Web Design & SEO Services in Dallas, TX | Webotixs',
    metaDescription: 'Webotixs provides professional web design, web development, SEO, hosting, website maintenance, AI chatbot, CRM and digital marketing services for businesses in Dallas, TX.',
    canonicalUrl: 'https://www.webotixs.com/locations/dallas-tx',
    eyebrow: 'WEBOTIXS — DALLAS, TX',
    h1: 'Web Design & Digital Services in Dallas, TX',
    heroSupport: 'Webotixs helps Dallas businesses build high-performance websites, improve their visibility on Google, automate customer interactions, and manage their digital operations with modern technology.',
    locationReferenceBadge: 'Serving Dallas-Fort Worth Metroplex Businesses',

    // Introduction
    introHeading: 'Digital Solutions for Businesses in Dallas, TX',
    introParagraphs: [
      'The Dallas-Fort Worth metroplex is home to one of the most dynamic business ecosystems in North America. From fast-growing corporate headquarters in Downtown Dallas to innovative enterprises across North Texas, competition for digital market share is fiercer than ever.',
      'To stand out in the Dallas commercial landscape, your business requires more than a simple online brochure. You need a modern digital platform engineered for lightning-fast loading speeds, frictionless user experiences, robust search engine visibility, and seamless customer lead conversion.',
      'Webotixs partners with Dallas companies to architect scalable web applications, implement revenue-focused local SEO campaigns, and integrate intelligent business automation. Whether you are expanding an established commercial brand or launching an ambitious new enterprise in Dallas, we engineer custom technology tailored to your growth objectives.',
    ],

    // Services
    servicesHeading: 'Our Digital Services in Dallas, TX',
    servicesSubheading: 'Full-spectrum web engineering, organic search optimization, and business automation for Dallas enterprises.',
    services: [
      {
        title: 'Web Design',
        description: 'Custom responsive web design crafted to engage Dallas audiences and turn site traffic into qualified leads.',
        icon: 'Palette',
        href: '/services/web-design-development',
      },
      {
        title: 'Web Development',
        description: 'Scalable Next.js, React, and custom web applications engineered for speed, security, and enterprise growth.',
        icon: 'Globe',
        href: '/services/web-design-development',
      },
      {
        title: 'SEO Services',
        description: 'Data-driven organic search strategies designed to elevate your brand to the top of Google search results.',
        icon: 'TrendingUp',
        href: '/services/seo-digital-marketing',
      },
      {
        title: 'Local SEO',
        description: 'Targeted local search and Google Business Profile optimization to dominate high-intent searches across Dallas.',
        icon: 'MapPin',
        href: '/services/seo-digital-marketing',
      },
      {
        title: 'Website Hosting',
        description: 'High-speed cloud infrastructure with automated daily backups, SSL encryption, and round-the-clock server monitoring.',
        icon: 'Cloud',
        href: '/services/cloud-devops',
      },
      {
        title: 'Website Maintenance',
        description: 'Proactive monthly website maintenance, security patching, core updates, and technical troubleshooting.',
        icon: 'ShieldCheck',
        href: '/services/cloud-devops',
      },
      {
        title: 'AI Chatbot Services',
        description: 'Custom AI chatbot development to capture leads, answer FAQs, and provide 24/7 automated support.',
        icon: 'Bot',
        href: '/services/web-design-development',
      },
      {
        title: 'CRM Solutions',
        description: 'Tailored CRM integration and custom pipeline development to manage leads and automate client communication.',
        icon: 'Database',
        href: '/services/web-design-development',
      },
      {
        title: 'Digital Marketing',
        description: 'Comprehensive digital marketing services engineered to amplify your reach and maximize digital ROI.',
        icon: 'Zap',
        href: '/services/seo-digital-marketing',
      },
      {
        title: 'WordPress Development',
        description: 'Custom WordPress development services providing easy content management without sacrificing speed or security.',
        icon: 'Code',
        href: '/services/web-design-development',
      },
      {
        title: 'Ecommerce Development',
        description: 'High-converting online stores built on Shopify and Next.js with secure payment gateways and fast checkout.',
        icon: 'ShoppingCart',
        href: '/services/ecommerce-solutions',
      },
      {
        title: 'UI/UX Design',
        description: 'Intuitive user interface and user experience design that aligns visual elegance with user retention goals.',
        icon: 'Layout',
        href: '/services/brand-identity-design',
      },
    ],

    // Web Design & Development
    webDevHeading: 'Web Design & Development Services in Dallas, TX',
    webDevCopy: [
      'Your website is the single most important digital touchpoint for your Dallas business. A slow, outdated, or confusing site actively drives prospective clients straight to your competitors.',
      'Our team delivers specialized Web Design Services Dallas TX and Web Development Services Dallas TX that combine aesthetic sophistication with cutting-edge engineering. We specialize in building mobile-first, conversion-focused digital platforms utilizing modern frameworks like Next.js, React, Tailwind CSS, and custom WordPress architectures.',
      'From custom Website Design Dallas TX projects to complex web applications and head-less ecommerce builds, Webotixs ensures your digital assets load instantly, look stunning on every screen size, and provide a seamless pathway from visitor curiosity to customer conversion.',
    ],
    webDevFeatures: [
      'Responsive Mobile-First Architecture',
      'Conversion-Optimized UX & Visual Hierarchy',
      'WordPress & Custom Next.js / React Solutions',
      'Core Web Vitals & Sub-Second Page Speeds',
      'Enterprise Ecommerce Platform Integration',
      'Clean, Modular & Maintainable Codebase',
    ],

    // SEO Section
    seoHeading: 'SEO Services in Dallas, TX',
    seoCopy: [
      'Appearing on the first page of search results for high-intent keywords in North Texas requires a strategic, analytical approach. Our SEO Services Dallas TX strategies are built to earn long-term authority and drive qualified organic search traffic.',
      'As a dedicated SEO Company Dallas TX, Webotixs executes deep technical site audits, comprehensive keyword research, structured schema implementation, and content optimization tailored to how Dallas customers evaluate services.',
      'Through focused Local SEO Services Dallas TX, we optimize your Google Business Profile, geographical relevance signals, and local landing pages to ensure your business dominates relevant searches across the Dallas-Fort Worth area.',
    ],
    seoCapabilities: [
      'Technical SEO Audits & On-Page Optimization',
      'Local SEO & Google Business Profile Strategy',
      'Targeted Keyword Research & Competitor Analysis',
      'Schema Markup & Rich Snippet Implementation',
      'High-Authority Content Strategy & Optimization',
      'Transparent Monthly Analytics & Ranking Reports',
    ],

    // Hosting & Maintenance
    hostingHeading: 'Reliable Website Hosting & Maintenance in Dallas, TX',
    hostingCopy: [
      'A successful website requires ongoing technical care, security updates, and high-speed infrastructure to maintain peak performance.',
      'Webotixs offers comprehensive Website Hosting Services Dallas TX and Website Maintenance Services Dallas TX designed to keep your site fast, secure, and operational around the clock.',
      'Our Monthly Website Maintenance Dallas TX packages include proactive security monitoring, software and plugin updates, automated daily backups, speed optimizations, and priority technical support whenever you need assistance.',
    ],
    hostingFeatures: [
      'Managed High-Speed Cloud Infrastructure',
      'Proactive Security Scanning & Firewall Protection',
      'Automated Daily Off-Site Backups',
      'Core, Theme, and Plugin Updates',
      'Continuous Uptime & Performance Monitoring',
      'Dedicated Technical Support & Bug Resolution',
    ],

    // AI Chatbot & Automation
    aiHeading: 'AI Chatbot & Business Automation in Dallas, TX',
    aiCopy: [
      'Modern consumers expect immediate responses to their inquiries, regardless of the time of day. Delayed responses often mean lost business opportunities.',
      'Webotixs provides specialized AI Chatbot Services Dallas TX and AI Chatbot Development Dallas TX to help Dallas companies automate customer support, capture visitor leads, and answer frequent questions instantly.',
      'Through AI Automation Services Dallas TX, we integrate smart conversational agents into your website, connecting them with your internal workflows and CRM databases to qualify leads automatically before routing them to your sales team.',
    ],
    aiUseCases: [
      '24/7 Lead Capture & Automated Qualification',
      'Instant FAQ & Technical Support Responses',
      'Automated Consultation & Appointment Booking',
      'Seamless Website & CRM Data Integration',
      'Custom Trained Knowledge Base Conversational AI',
      'Multi-Channel Messaging & Notification Flows',
    ],

    // CRM Services
    crmHeading: 'CRM Solutions for Growing Dallas Businesses',
    crmCopy: [
      'Managing customer relationships efficiently is essential for sustainable growth. Without a central system, leads slip through the cracks and customer communications become fragmented.',
      'Our CRM Services Dallas TX and CRM Development Dallas TX solutions empower Dallas businesses to track leads from initial contact to closed deal. We design custom CRM architectures and integrate leading platforms tailored to your sales process.',
      'With Custom CRM Dallas TX integrations, your team gains real-time visibility into sales pipelines, automated follow-up sequences, task assignments, and comprehensive performance analytics.',
    ],
    crmFeatures: [
      'Centralized Lead & Contact Management',
      'Custom Visual Sales Pipeline Workflows',
      'Automated Email & Task Follow-Up Sequences',
      'Custom Analytics & Executive Reporting Dashboards',
      'Third-Party Application & API Integrations',
      'Role-Based Team Access & Security Controls',
    ],

    // Local Relevance
    localAreaHeading: 'Serving Businesses Across Dallas & the Surrounding Area',
    localAreaIntro: 'Webotixs is proud to support businesses, startups, and commercial enterprises throughout the Dallas-Fort Worth metroplex and surrounding North Texas communities, including:',
    localAreas: [
      'Dallas',
      'Plano',
      'Irving',
      'Richardson',
      'Frisco',
      'Garland',
      'McKinney',
      'Arlington',
    ],

    // FAQs
    faqsHeading: 'Frequently Asked Questions (Dallas, TX)',
    faqs: [
      {
        question: 'What web design and digital services does Webotixs offer in Dallas, TX?',
        answer: 'Webotixs provides comprehensive web services for Dallas businesses, including custom web design, web development, local and organic SEO, high-speed website hosting, monthly website maintenance, AI chatbot development, custom CRM integration, e-commerce builds, and UI/UX design.',
      },
      {
        question: 'How long does it take to design and launch a new website for a Dallas business?',
        answer: 'Project timelines depend on the scope and complexity of the website. A standard custom business website typically takes 3 to 5 weeks from discovery to launch, while complex enterprise platforms or custom web applications may take 6 to 10 weeks.',
      },
      {
        question: 'How does Webotixs help Dallas companies improve their Google search rankings?',
        answer: 'We execute data-driven SEO strategies including technical site audits, keyword optimization, Google Business Profile enhancement, high-quality content creation, and local schema implementation to increase your visibility for high-intent search queries in Dallas.',
      },
      {
        question: 'Does Webotixs offer ongoing monthly website maintenance in Dallas?',
        answer: 'Yes. Our monthly website maintenance plans include continuous security monitoring, regular software updates, daily automated backups, performance tuning, and technical support to ensure your website remains fast and secure.',
      },
      {
        question: 'Can Webotixs build custom AI chatbots and CRM solutions for my business?',
        answer: 'Absolutely. We develop custom AI chatbots that handle 24/7 visitor lead capture and support inquiries, as well as CRM solutions that automate sales pipelines and organize lead workflows for your growth.',
      },
      {
        question: 'How do I get started with a web design or SEO project in Dallas?',
        answer: 'You can reach out to our team by clicking "Start Your Project" or filling out our contact form. We will arrange an initial consultation to review your goals and provide a detailed technical proposal within 24 hours.',
      },
    ],

    // Final CTA
    ctaHeading: 'Ready to Grow Your Business in Dallas?',
    ctaSupport: 'Build a faster, smarter and more effective digital presence with Webotixs.',
  },

  'las-vegas-nv': {
    city: 'Las Vegas',
    state: 'Nevada',
    stateAbbr: 'NV',
    slug: 'las-vegas-nv',
    pageTitle: 'Web Design & SEO Services in Las Vegas, NV | Webotixs',
    metaDescription: 'Webotixs provides professional web design, web development, SEO, hosting, website maintenance, AI chatbot, CRM and digital marketing services for businesses in Las Vegas, NV.',
    canonicalUrl: 'https://www.webotixs.com/locations/las-vegas-nv',
    eyebrow: 'WEBOTIXS — LAS VEGAS, NV',
    h1: 'Web Design & Digital Services in Las Vegas, NV',
    heroSupport: 'Webotixs helps Las Vegas businesses build high-performance websites, improve their visibility on Google, automate customer interactions, and manage their digital operations with modern technology.',
    locationReferenceBadge: 'Serving Las Vegas Valley & Clark County Enterprises',

    // Introduction
    introHeading: 'Digital Solutions for Businesses in Las Vegas, NV',
    introParagraphs: [
      'Las Vegas is one of the world’s most energetic, fast-moving economic hubs. Beyond the world-famous Entertainment Capital of the World lies a flourishing ecosystem of professional services, tech innovations, commercial real estate, health services, and specialized retail businesses.',
      'Operating in Southern Nevada requires a strong, distinctive online presence. Las Vegas consumers and visiting decision-makers expect 24/7 accessibility, instant page load speeds, intuitive mobile navigation, and clear visual credibility.',
      'Webotixs equips Las Vegas businesses with modern web engineering, strategic SEO campaigns, and intelligent business automation. We design digital platforms built to capture attention, dominate regional search rankings, and convert digital interest into sustained revenue.',
    ],

    // Services
    servicesHeading: 'Our Digital Services in Las Vegas, NV',
    servicesSubheading: 'Comprehensive web design, search engine optimization, and custom digital automation for Las Vegas businesses.',
    services: [
      {
        title: 'Web Design',
        description: 'Visually compelling, high-converting web design customized to reflect the dynamic spirit of your Las Vegas brand.',
        icon: 'Palette',
        href: '/services/web-design-development',
      },
      {
        title: 'Web Development',
        description: 'Robust Next.js, React, and custom web application engineering optimized for maximum reliability and speed.',
        icon: 'Globe',
        href: '/services/web-design-development',
      },
      {
        title: 'SEO Services',
        description: 'Strategic search engine optimization designed to position your business at the top of organic search results.',
        icon: 'TrendingUp',
        href: '/services/seo-digital-marketing',
      },
      {
        title: 'Local SEO',
        description: 'Targeted local SEO strategies and Google Business Profile optimization to capture high-intent Las Vegas searches.',
        icon: 'MapPin',
        href: '/services/seo-digital-marketing',
      },
      {
        title: 'Website Hosting',
        description: 'Ultra-fast managed cloud hosting with SSL security, automated daily backups, and constant server uptime.',
        icon: 'Cloud',
        href: '/services/cloud-devops',
      },
      {
        title: 'Website Maintenance',
        description: 'Dependable monthly website maintenance, security auditing, performance optimizations, and prompt technical support.',
        icon: 'ShieldCheck',
        href: '/services/cloud-devops',
      },
      {
        title: 'AI Chatbot Services',
        description: 'Intelligent AI chatbot development to handle customer inquiries 24/7, schedule consultations, and capture leads.',
        icon: 'Bot',
        href: '/services/web-design-development',
      },
      {
        title: 'CRM Solutions',
        description: 'Tailored CRM development and integration to streamline sales management and client communication workflows.',
        icon: 'Database',
        href: '/services/web-design-development',
      },
      {
        title: 'Digital Marketing',
        description: 'Results-driven digital marketing campaigns engineered to expand brand visibility and maximize online ROI.',
        icon: 'Zap',
        href: '/services/seo-digital-marketing',
      },
      {
        title: 'WordPress Development',
        description: 'Custom WordPress development offering intuitive content administration paired with high-performance code.',
        icon: 'Code',
        href: '/services/web-design-development',
      },
      {
        title: 'Ecommerce Development',
        description: 'Scalable e-commerce solutions built on Shopify and Next.js designed for seamless shopping and secure checkout.',
        icon: 'ShoppingCart',
        href: '/services/ecommerce-solutions',
      },
      {
        title: 'UI/UX Design',
        description: 'Modern user interface and user experience design focused on visual clarity, engagement, and customer retention.',
        icon: 'Layout',
        href: '/services/brand-identity-design',
      },
    ],

    // Web Design & Development
    webDevHeading: 'Web Design & Development Services in Las Vegas, NV',
    webDevCopy: [
      'In a market as visually driven as Las Vegas, your website serves as your primary digital storefront. An outdated or poorly structured website diminishes brand trust and forfeits valuable opportunities to competitors.',
      'Our team delivers specialized Web Design Services Las Vegas NV and Web Development Services Las Vegas NV that blend sleek aesthetic design with high-performance modern web engineering. We build lightning-fast, mobile-optimized sites using frameworks like Next.js, React, Tailwind CSS, and custom WordPress environments.',
      'Whether you require custom Website Design Las Vegas NV or advanced Custom Web Development Las Vegas NV applications, Webotixs ensures your platform renders seamlessly across mobile devices, tablets, and desktops while delivering measurable business growth.',
    ],
    webDevFeatures: [
      'Sleek Mobile-First Responsive Design',
      'Conversion-Centric UX & Architecture',
      'WordPress & Custom Next.js / React Solutions',
      'Optimized Page Speeds & Core Web Vitals',
      'Custom Ecommerce & Headless Solutions',
      'Clean, Secure, and Future-Proof Codebase',
    ],

    // SEO Section
    seoHeading: 'SEO Services in Las Vegas, NV',
    seoCopy: [
      'Achieving top rankings on Google in Southern Nevada demands an authoritative, data-backed SEO approach. Our SEO Services Las Vegas NV strategies focus on capturing high-intent search queries from prospective clients.',
      'As an experienced SEO Company Las Vegas NV, Webotixs conducts comprehensive technical site optimizations, in-depth keyword analysis, structured schema implementation, and content enhancements aligned with user search behavior.',
      'Through targeted Local SEO Services Las Vegas NV, we optimize your Google Business Profile and local geo-targeted signals so your business ranks prominently when prospective customers search for your services in Las Vegas.',
    ],
    seoCapabilities: [
      'Comprehensive Technical SEO & On-Page Audits',
      'Local SEO & Google Business Profile Management',
      'Targeted Keyword Research & Intent Mapping',
      'Schema Markup & Structured Data Integration',
      'Search-Engine Optimized Content Strategy',
      'Transparent Monthly Analytics & Growth Tracking',
    ],

    // Hosting & Maintenance
    hostingHeading: 'Reliable Website Hosting & Maintenance in Las Vegas, NV',
    hostingCopy: [
      'Maintaining a competitive digital presence requires dependable infrastructure, ongoing security monitoring, and regular maintenance.',
      'Webotixs provides enterprise-grade Website Hosting Services Las Vegas NV and Website Maintenance Services Las Vegas NV to ensure your website remains fast, fully operational, and safe from cyber threats.',
      'Our Monthly Website Maintenance Las Vegas NV packages encompass routine security patching, continuous performance monitoring, automated daily backups, core software updates, and rapid technical assistance.',
    ],
    hostingFeatures: [
      'High-Performance Managed Cloud Hosting',
      'Proactive Firewall Security & Threat Scanning',
      'Automated Daily Off-Site Backups',
      'Regular Core, Theme, and Plugin Updates',
      'Real-Time Uptime & Speed Monitoring',
      'Dedicated Technical Support & Issue Resolution',
    ],

    // AI Chatbot & Automation
    aiHeading: 'AI Chatbot & Business Automation in Las Vegas, NV',
    aiCopy: [
      'In a 24-hour city like Las Vegas, inquiries arrive around the clock. Providing immediate, accurate responses is critical to securing new business.',
      'Webotixs offers specialized AI Chatbot Services Las Vegas NV and AI Chatbot Development Las Vegas NV to help Southern Nevada businesses automate lead capture, answer FAQs, and engage prospective customers instantly.',
      'Through AI Automation Services Las Vegas NV, we build intelligent conversational agents that integrate directly into your website and CRM, automating initial qualifications and routing high-value leads straight to your team.',
    ],
    aiUseCases: [
      '24/7 Automated Lead Capture & Qualification',
      'Instant Responses to Common Service Inquiries',
      'Automated Consultation & Appointment Booking',
      'Seamless Integration with CRM & Internal Tools',
      'Custom Knowledge Base Trained AI Assistant',
      'Multi-Channel Messaging & Notification Workflows',
    ],

    // CRM Services
    crmHeading: 'CRM Solutions for Growing Las Vegas Businesses',
    crmCopy: [
      'Organizing customer data and sales pipelines effectively is vital for scaling your operations. Disorganized systems lead to delayed follow-ups and lost revenue.',
      'Our CRM Services Las Vegas NV and CRM Development Las Vegas NV solutions equip Las Vegas organizations to manage client relationships systematically from initial contact through service delivery.',
      'With Custom CRM Las Vegas NV setups, your business gains clear visibility into sales pipelines, automated task reminders, client activity tracking, and real-time executive performance dashboards.',
    ],
    crmFeatures: [
      'Centralized Customer & Lead Management',
      'Custom Visual Sales & Opportunity Pipelines',
      'Automated Lead Nurturing & Follow-Up Reminders',
      'Custom Reporting & Analytics Dashboards',
      'API Integration with Existing Software Stacks',
      'Secure Multi-User Access & Role Controls',
    ],

    // Local Relevance
    localAreaHeading: 'Serving Businesses Across Las Vegas & the Surrounding Area',
    localAreaIntro: 'Webotixs delivers custom digital solutions to companies, commercial brands, and growing enterprises across the Las Vegas Valley and surrounding Southern Nevada communities, including:',
    localAreas: [
      'Las Vegas',
      'Henderson',
      'North Las Vegas',
      'Summerlin',
      'Enterprise',
    ],

    // FAQs
    faqsHeading: 'Frequently Asked Questions (Las Vegas, NV)',
    faqs: [
      {
        question: 'What web design and digital services does Webotixs provide in Las Vegas, NV?',
        answer: 'Webotixs offers a full range of digital services for Las Vegas companies, including custom web design, web development, local and organic SEO, high-speed cloud hosting, monthly website maintenance, AI chatbot development, CRM integration, e-commerce development, and UI/UX design.',
      },
      {
        question: 'How long does a web design project take for a Las Vegas business?',
        answer: 'Timelines vary based on requirements. A standard custom business website generally takes 3 to 5 weeks from discovery to deployment. More extensive web applications or custom e-commerce builds typically require 6 to 10 weeks.',
      },
      {
        question: 'How can Webotixs improve our Google search rankings in Las Vegas?',
        answer: 'We deploy comprehensive SEO strategies tailored for Las Vegas, including technical on-page audits, targeted keyword research, Google Business Profile optimization, local link building signals, and high-authority content creation.',
      },
      {
        question: 'Does Webotixs offer website maintenance plans in Las Vegas?',
        answer: 'Yes. Our monthly website maintenance plans provide proactive security scanning, regular software updates, daily automated backups, page speed optimizations, and ongoing technical support to keep your site operating flawlessly.',
      },
      {
        question: 'Can Webotixs build an AI chatbot or custom CRM for our organization?',
        answer: 'Yes. We engineer custom AI chatbots for 24/7 lead qualification and customer support, as well as CRM solutions that automate sales pipelines and streamline client management.',
      },
      {
        question: 'How can we start a project with Webotixs in Las Vegas?',
        answer: 'Simply click "Start Your Project" or complete our online contact form. Our team will contact you to discuss your objectives and deliver a detailed technical roadmap within 24 hours.',
      },
    ],

    // Final CTA
    ctaHeading: 'Ready to Grow Your Business in Las Vegas?',
    ctaSupport: 'Build a faster, smarter and more effective digital presence with Webotixs.',
  },
}
