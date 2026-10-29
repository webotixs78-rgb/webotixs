const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

// 1200 x 630 dimensions (1.91:1 aspect ratio)
const WIDTH = 1200;
const HEIGHT = 630;

const svg = `
<svg width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Dark Space/Slate Background Gradient -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#080C14"/>
      <stop offset="50%" stop-color="#0B101E"/>
      <stop offset="100%" stop-color="#0D1527"/>
    </linearGradient>

    <!-- Brand Cyan-Blue Primary Gradient -->
    <linearGradient id="primaryGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2563EB"/>
      <stop offset="100%" stop-color="#06B6D4"/>
    </linearGradient>

    <!-- Glow/Text Gradient -->
    <linearGradient id="textGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="50%" stop-color="#06B6D4"/>
      <stop offset="100%" stop-color="#3B82F6"/>
    </linearGradient>

    <!-- Glass Fill -->
    <linearGradient id="glassGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="rgba(255, 255, 255, 0.08)"/>
      <stop offset="100%" stop-color="rgba(255, 255, 255, 0.02)"/>
    </linearGradient>

    <!-- Card Fill -->
    <linearGradient id="cardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#111827"/>
      <stop offset="100%" stop-color="#0E1626"/>
    </linearGradient>

    <!-- Glowing Radial Spheres -->
    <radialGradient id="sphere1" cx="20%" cy="15%" r="65%">
      <stop offset="0%" stop-color="#2563EB" stop-opacity="0.32"/>
      <stop offset="60%" stop-color="#2563EB" stop-opacity="0.08"/>
      <stop offset="100%" stop-color="#2563EB" stop-opacity="0"/>
    </radialGradient>

    <radialGradient id="sphere2" cx="80%" cy="40%" r="55%">
      <stop offset="0%" stop-color="#06B6D4" stop-opacity="0.28"/>
      <stop offset="60%" stop-color="#06B6D4" stop-opacity="0.05"/>
      <stop offset="100%" stop-color="#06B6D4" stop-opacity="0"/>
    </radialGradient>

    <radialGradient id="sphere3" cx="50%" cy="85%" r="50%">
      <stop offset="0%" stop-color="#8B5CF6" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#8B5CF6" stop-opacity="0"/>
    </radialGradient>

    <!-- Subtle Grid Pattern -->
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255, 255, 255, 0.03)" stroke-width="1"/>
    </pattern>

    <!-- Filter for soft glows -->
    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="12" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>

    <filter id="cardShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="16" stdDeviation="24" flood-color="#000000" flood-opacity="0.6"/>
    </filter>
  </defs>

  <!-- 1. Background -->
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#bgGrad)"/>

  <!-- 2. Subtle Tech Grid -->
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#grid)"/>

  <!-- 3. Ambient Lighting Glow Orbs -->
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#sphere1)"/>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#sphere2)"/>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#sphere3)"/>

  <!-- Subtle Top Decorative Glow Line -->
  <rect x="0" y="0" width="${WIDTH}" height="3" fill="url(#primaryGrad)"/>

  <!-- ==================== LEFT COLUMN: BRAND & VALUE PROPOSITION ==================== -->

  <!-- Webotixs Logo & Brand Lockup -->
  <g transform="translate(80, 75)">
    <!-- Brand Icon Container -->
    <rect x="0" y="0" width="56" height="56" rx="14" fill="url(#primaryGrad)"/>
    <rect x="2" y="2" width="52" height="52" rx="12" fill="#0B101E"/>
    <!-- Logo "W" Gradient Text / Symbol -->
    <text x="28" y="37" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-weight="900" font-size="28" fill="url(#textGrad)" text-anchor="middle">W</text>
    
    <!-- Brand Typography -->
    <text x="72" y="38" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-weight="800" font-size="34" fill="#FFFFFF" letter-spacing="-0.5">Webotixs</text>
    
    <!-- Verified Quality Badge -->
    <rect x="235" y="16" width="128" height="26" rx="13" fill="rgba(6, 182, 212, 0.12)" stroke="rgba(6, 182, 212, 0.3)" stroke-width="1"/>
    <circle cx="248" cy="29" r="4" fill="#10B981"/>
    <text x="258" y="33" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-weight="600" font-size="11" fill="#38BDF8" letter-spacing="0.5">DIGITAL AGENCY</text>
  </g>

  <!-- Main Headline -->
  <g transform="translate(80, 195)">
    <text font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="52" font-weight="800" fill="#FFFFFF" letter-spacing="-1.2">
      <tspan x="0" y="0">We Build Digital</tspan>
      <tspan x="0" y="64" fill="url(#textGrad)">Experiences</tspan>
      <tspan fill="#FFFFFF"> That Convert</tspan>
    </text>
  </g>

  <!-- Value Proposition Description -->
  <g transform="translate(80, 355)">
    <text font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="19" font-weight="400" fill="#94A3B8" letter-spacing="-0.2">
      <tspan x="0" y="0">High-performance websites, AI automations, and custom CRM</tspan>
      <tspan x="0" y="28">solutions engineered to scale ambitious businesses.</tspan>
    </text>
  </g>

  <!-- Service Capability Pills -->
  <g transform="translate(80, 435)">
    <!-- Pill 1: Web Design -->
    <g transform="translate(0, 0)">
      <rect width="138" height="36" rx="10" fill="url(#glassGrad)" stroke="rgba(255, 255, 255, 0.1)" stroke-width="1"/>
      <circle cx="16" cy="18" r="3.5" fill="#38BDF8"/>
      <text x="28" y="23" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="13" font-weight="600" fill="#E2E8F0">Web Design</text>
    </g>

    <!-- Pill 2: AI Automation -->
    <g transform="translate(150, 0)">
      <rect width="150" height="36" rx="10" fill="url(#glassGrad)" stroke="rgba(255, 255, 255, 0.1)" stroke-width="1"/>
      <circle cx="16" cy="18" r="3.5" fill="#06B6D4"/>
      <text x="28" y="23" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="13" font-weight="600" fill="#E2E8F0">AI Automation</text>
    </g>

    <!-- Pill 3: Custom CRM -->
    <g transform="translate(312, 0)">
      <rect width="180" height="36" rx="10" fill="url(#glassGrad)" stroke="rgba(255, 255, 255, 0.1)" stroke-width="1"/>
      <circle cx="16" cy="18" r="3.5" fill="#818CF8"/>
      <text x="28" y="23" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="13" font-weight="600" fill="#E2E8F0">Custom CRM Solutions</text>
    </g>
  </g>

  <!-- Website URL & Metrics Footer -->
  <g transform="translate(80, 528)">
    <!-- Globe Icon -->
    <g transform="translate(0, -14)">
      <circle cx="12" cy="12" r="11" fill="none" stroke="#06B6D4" stroke-width="1.8"/>
      <path d="M 12 1 C 8 7, 8 17, 12 23 C 16 17, 16 7, 12 1" fill="none" stroke="#06B6D4" stroke-width="1.4"/>
      <line x1="2" y1="12" x2="22" y2="12" stroke="#06B6D4" stroke-width="1.5"/>
    </g>
    <text x="34" y="2" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="18" font-weight="700" fill="#38BDF8" letter-spacing="0.4">www.webotixs.com</text>

    <!-- Social Proof Rating -->
    <g transform="translate(310, -14)">
      <rect width="190" height="28" rx="8" fill="rgba(255, 255, 255, 0.04)" stroke="rgba(255, 255, 255, 0.08)" stroke-width="1"/>
      <!-- 5 Stars -->
      <g transform="translate(10, 6)">
        <polygon points="6,0 7.8,4.5 12,4.8 8.8,7.8 9.8,12 6,9.5 2.2,12 3.2,7.8 0,4.8 4.2,4.5" fill="#FBBF24"/>
        <polygon points="20,0 21.8,4.5 26,4.8 22.8,7.8 23.8,12 20,9.5 16.2,12 17.2,7.8 14,4.8 18.2,4.5" fill="#FBBF24"/>
        <polygon points="34,0 35.8,4.5 40,4.8 36.8,7.8 37.8,12 34,9.5 30.2,12 31.2,7.8 28,4.8 32.2,4.5" fill="#FBBF24"/>
        <polygon points="48,0 49.8,4.5 54,4.8 50.8,7.8 51.8,12 48,9.5 44.2,12 45.2,7.8 42,4.8 46.2,4.5" fill="#FBBF24"/>
        <polygon points="62,0 63.8,4.5 68,4.8 64.8,7.8 65.8,12 62,9.5 58.2,12 59.2,7.8 56,4.8 60.2,4.5" fill="#FBBF24"/>
      </g>
      <text x="86" y="19" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="12" font-weight="700" fill="#F8FAFC">5.0 Star Agency</text>
    </g>
  </g>


  <!-- ==================== RIGHT COLUMN: MODERN TECH & UI MOCKUP ==================== -->

  <!-- Main Showcase Device / Card (Perspective Tilt Look) -->
  <g transform="translate(670, 75)" filter="url(#cardShadow)">
    <!-- Base Card Container -->
    <rect width="450" height="475" rx="20" fill="url(#cardGrad)" stroke="rgba(255, 255, 255, 0.12)" stroke-width="1.2"/>
    
    <!-- Browser Top Bar -->
    <rect width="450" height="42" rx="20" fill="#161F30"/>
    <rect y="22" width="450" height="20" fill="#161F30"/>
    
    <!-- Traffic Light Window Dots -->
    <circle cx="24" cy="21" r="5" fill="#EF4444"/>
    <circle cx="40" cy="21" r="5" fill="#F59E0B"/>
    <circle cx="56" cy="21" r="5" fill="#10B981"/>

    <!-- URL Bar in Mockup -->
    <rect x="80" y="10" width="290" height="22" rx="6" fill="#0B101E" stroke="rgba(255, 255, 255, 0.08)" stroke-width="1"/>
    <text x="225" y="25" font-family="'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif" font-size="10" font-weight="500" fill="#64748B" text-anchor="middle">webotixs.com/solutions</text>

    <!-- Inner Application Content -->
    <!-- Hero Mockup Header inside card -->
    <g transform="translate(24, 62)">
      <!-- Mini Brand Tag -->
      <rect width="90" height="18" rx="6" fill="rgba(37, 99, 235, 0.2)" stroke="rgba(37, 99, 235, 0.4)" stroke-width="0.8"/>
      <text x="45" y="13" font-family="'Segoe UI', sans-serif" font-size="9" font-weight="700" fill="#60A5FA" text-anchor="middle">NEXT.JS 16 ARCHITECTURE</text>
      
      <!-- Mini Headline Mockup -->
      <rect x="0" y="26" width="260" height="12" rx="4" fill="#FFFFFF" fill-opacity="0.9"/>
      <rect x="0" y="44" width="180" height="10" rx="3" fill="url(#textGrad)"/>
      <rect x="0" y="60" width="220" height="7" rx="3" fill="#475569"/>
    </g>

    <!-- Mini Dashboard Stats Cards in Mockup -->
    <g transform="translate(24, 145)">
      <!-- Stat Card 1 -->
      <g transform="translate(0, 0)">
        <rect width="125" height="74" rx="12" fill="rgba(255, 255, 255, 0.03)" stroke="rgba(255, 255, 255, 0.07)" stroke-width="1"/>
        <text x="14" y="24" font-family="'Segoe UI', sans-serif" font-size="10" font-weight="600" fill="#94A3B8">Conversion Rate</text>
        <text x="14" y="52" font-family="'Segoe UI', sans-serif" font-size="22" font-weight="800" fill="#10B981">+180%</text>
      </g>
      <!-- Stat Card 2 -->
      <g transform="translate(138, 0)">
        <rect width="125" height="74" rx="12" fill="rgba(255, 255, 255, 0.03)" stroke="rgba(255, 255, 255, 0.07)" stroke-width="1"/>
        <text x="14" y="24" font-family="'Segoe UI', sans-serif" font-size="10" font-weight="600" fill="#94A3B8">Page Speed</text>
        <text x="14" y="52" font-family="'Segoe UI', sans-serif" font-size="22" font-weight="800" fill="#38BDF8">99/100</text>
      </g>
      <!-- Stat Card 3 -->
      <g transform="translate(276, 0)">
        <rect width="125" height="74" rx="12" fill="rgba(255, 255, 255, 0.03)" stroke="rgba(255, 255, 255, 0.07)" stroke-width="1"/>
        <text x="14" y="24" font-family="'Segoe UI', sans-serif" font-size="10" font-weight="600" fill="#94A3B8">Uptime</text>
        <text x="14" y="52" font-family="'Segoe UI', sans-serif" font-size="22" font-weight="800" fill="#A855F7">99.9%</text>
      </g>
    </g>

    <!-- Visual Chart / Performance Graph Line -->
    <g transform="translate(24, 240)">
      <rect width="402" height="110" rx="12" fill="rgba(15, 23, 42, 0.6)" stroke="rgba(255, 255, 255, 0.06)" stroke-width="1"/>
      <text x="16" y="24" font-family="'Segoe UI', sans-serif" font-size="11" font-weight="600" fill="#CBD5E1">Traffic &amp; Revenue Growth</text>
      
      <!-- Graph path -->
      <path d="M 20 86 Q 80 80, 140 60 T 260 45 T 380 18" fill="none" stroke="url(#textGrad)" stroke-width="3" stroke-linecap="round"/>
      <path d="M 20 86 Q 80 80, 140 60 T 260 45 T 380 18 L 380 95 L 20 95 Z" fill="url(#sphere2)" opacity="0.3"/>
      
      <!-- Indicator Dot -->
      <circle cx="380" cy="18" r="5" fill="#38BDF8" stroke="#FFFFFF" stroke-width="2"/>
    </g>

    <!-- Mini Bottom CTAs -->
    <g transform="translate(24, 370)">
      <rect width="180" height="34" rx="10" fill="url(#primaryGrad)"/>
      <text x="90" y="22" font-family="'Segoe UI', sans-serif" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">Launch Project</text>
      
      <rect x="195" y="0" width="130" height="34" rx="10" fill="rgba(255, 255, 255, 0.05)" stroke="rgba(255, 255, 255, 0.1)" stroke-width="1"/>
      <text x="260" y="22" font-family="'Segoe UI', sans-serif" font-size="12" font-weight="600" fill="#94A3B8" text-anchor="middle">View Portfolio</text>
    </g>

    <!-- Tech Stack Icons in Footer of Mockup -->
    <g transform="translate(24, 430)">
      <text x="0" y="16" font-family="'Segoe UI', sans-serif" font-size="10" font-weight="600" fill="#475569">POWERED BY NEXT.JS 16 • REACT 19 • SUPABASE • TAILWIND CSS</text>
    </g>
  </g>

  <!-- Floating Glass Tag 1 (Top Right) -->
  <g transform="translate(980, 45)" filter="url(#cardShadow)">
    <rect width="150" height="44" rx="12" fill="#0B132B" stroke="#06B6D4" stroke-width="1.2"/>
    <circle cx="22" cy="22" r="5" fill="#10B981"/>
    <text x="36" y="27" font-family="'Segoe UI', sans-serif" font-size="13" font-weight="700" fill="#FFFFFF">AI-Powered</text>
  </g>

  <!-- Floating Glass Tag 2 (Bottom Right) -->
  <g transform="translate(620, 480)" filter="url(#cardShadow)">
    <rect width="190" height="48" rx="14" fill="#0E172A" stroke="rgba(56, 189, 248, 0.4)" stroke-width="1.2"/>
    <text x="20" y="22" font-family="'Segoe UI', sans-serif" font-size="10" font-weight="600" fill="#94A3B8">ENTERPRISE GRADE</text>
    <text x="20" y="38" font-family="'Segoe UI', sans-serif" font-size="13" font-weight="800" fill="#38BDF8">Full-Stack Scalability</text>
  </g>

</svg>
`;

async function generate() {
  const publicDir = path.join(__dirname, '..', 'public');
  const svgPath = path.join(publicDir, 'og-webotixs.svg');
  const jpgPath = path.join(publicDir, 'og-webotixs.jpg');
  const pngPath = path.join(publicDir, 'og-webotixs.png');
  const oldJpgPath = path.join(publicDir, 'og-image.jpg');

  // Save the SVG source
  fs.writeFileSync(svgPath, svg.trim());
  console.log('Saved SVG to:', svgPath);

  // Render High Quality JPG (1200x630, quality 95)
  await sharp(Buffer.from(svg))
    .jpeg({ quality: 95, chromaSubsampling: '4:4:4' })
    .toFile(jpgPath);
  console.log('Generated JPG to:', jpgPath);

  // Also render PNG version
  await sharp(Buffer.from(svg))
    .png({ compressionLevel: 8 })
    .toFile(pngPath);
  console.log('Generated PNG to:', pngPath);

  // Also update og-image.jpg so any existing references immediately benefit
  await sharp(Buffer.from(svg))
    .jpeg({ quality: 95, chromaSubsampling: '4:4:4' })
    .toFile(oldJpgPath);
  console.log('Generated fallback og-image.jpg to:', oldJpgPath);

  const jpgStats = fs.statSync(jpgPath);
  const pngStats = fs.statSync(pngPath);
  console.log(`Dimensions: ${WIDTH}x${HEIGHT}`);
  console.log(`JPG Size: ${(jpgStats.size / 1024).toFixed(1)} KB`);
  console.log(`PNG Size: ${(pngStats.size / 1024).toFixed(1)} KB`);
}

generate().catch(console.error);
