# Webotixs — Award-Winning Enterprise Web Design & Digital Innovation Agency Platform

An ultra-modern, high-performance, award-winning web design and digital agency platform inspired by HuxenTech workflows while featuring a **completely original, futuristic dark-mode aesthetic** and **production-ready codebase**.

Built with **Next.js 16 (App Router + Turbopack)**, **React 19**, **TypeScript**, **Tailwind CSS v4**, **Framer Motion**, **GSAP**, **Supabase PostgreSQL / Storage / Auth**, **Zod**, **Tiptap Rich Text Editor**, **OpenAI GPT-4o-mini Lead Classification**, and **Resend Email Dispatches**.

---

## ✨ Key Features & Highlights

### 🎨 Visual & Frontend Excellence
- **Vibrant Dark-Theme Design System**: Curated HSL glassmorphism (`glass`), deep obsidian backgrounds (`#050816`), neon mesh gradients, and smooth glowing card borders (`shadow-glow-sm`).
- **Interactive Animations & Micro-Interactions**:
  - `ParticleBackground`: 60fps HTML5 Canvas connected node network responding to mouse movement.
  - `CustomCursor`: Lagging glowing ring cursor with auto-scale hover states over clickable elements.
  - `ScrollReveal`: Direction-aware scroll-triggered fade animations.
  - `AnimatedCounter`: Dynamic view-triggered number easing.
- **Full Internationalization (i18n + RTL)**: Live language selector supporting **English (`EN`)**, **Arabic (`AR` — العربية)** with dynamic `dir="rtl"`, **Chinese (`ZH` — 中文)**, and **Urdu (`UR` — اردو)**.
- **18+ Public Pages**: Home, About, Services Catalog & Detail (`/services/[slug]`), Portfolio Grid & Case Studies (`/portfolio/[slug]`), Industries Verticals, Team Profiles, Blog Grid & Full Articles (`/blog/[slug]`), Careers & Perks, Contact Form (`React Hook Form` + `Zod`), Client Search (`/search`), and Legal/Privacy docs.

### 🛠️ Full-Stack Admin CMS & CRM Dashboard (`/admin`)
Protected by Next.js 16 `proxy.ts` (modern proxy convention replacing deprecated middleware) and Supabase Authentication (`/admin/login`).

- **Dashboard Overview**: Key metrics, recent inquiries, quick action shortcuts, and active server health status.
- **Services Manager**: Full CRUD table and modal editor with custom feature tags and multi-step process workflows.
- **Portfolio Manager Grid**: Manage client case studies, challenge/solution summaries, technology tags, and key quantifiable metric cards (`320% ROI`, `99.99% Uptime`).
- **Blog Manager with Tiptap WYSIWYG Editor**: Write, format, tag, and categorize technical articles and thought leadership posts using an embedded Tiptap editor with headings, lists, code blocks, and images.
- **Media Library Bucket Manager (`/admin/media`)**: Organize files into virtual folders (`Backgrounds`, `Portfolio`, `Team`, `Documents`, `Videos`), toggle grid/list view, copy public CDN URLs with one click, and upload directly to Supabase Storage.
- **Careers & Job Application Evaluation Pipeline (`/admin/careers`)**: Create job openings and evaluate applicant resumes/cover letters through stage-based pipelines (`new` → `reviewing` → `shortlisted` → `hired`).
- **AI-Powered CRM Lead Pipeline (`/admin/crm`)**: Receives contact form submissions and utilizes **OpenAI GPT-4o-mini** to automatically classify lead priority (`high`, `medium`, `low`) and generate concise 2-sentence executive summaries.
- **Audit Logs Manager (`/admin/logs`)**: Real-time activity audit trail tracking admin logins, entity creation, edits, and deletions.
- **Backup & Export Utilities (`/admin/backup`)**: One-click JSON snapshot exports for individual modules or full database bundles, plus local factory seed restore options.
- **Site Configuration Settings Panel (`/admin/settings`)**: Global brand colors, social media links, corporate addresses, and analytics tracker keys.

---

## 🏗️ Technology Stack

| Layer | Technology | Version / Purpose |
|---|---|---|
| **Core Framework** | [Next.js](https://nextjs.org/) | `16.2.10` (App Router, Server Actions, Turbopack, Proxy) |
| **UI Library** | [React](https://react.dev/) | `19.2.4` (React Server Components, Hooks) |
| **Language** | [TypeScript](https://www.typescriptlang.org/) | `5.x` Strict type safety across client and database models |
| **Styling** | [Tailwind CSS](https://tailwindcss.com/) | `v4` with custom CSS variables (`globals.css`) & PostCSS |
| **Motion & Animation**| [Framer Motion](https://www.framer.com/motion/) & [GSAP](https://gsap.com/) | `12.x` / `3.15.0` High-performance layout transitions and timeline animations |
| **Database & Auth** | [Supabase](https://supabase.com/) | `@supabase/ssr` (`0.12.3`) & `supabase-js` (`2.110.6`) PostgreSQL & RLS |
| **Rich Text Editor** | [Tiptap](https://tiptap.dev/) | `@tiptap/react` (`3.27.4`) + Starter Kit, Image, Link extensions |
| **AI Lead Engine** | [OpenAI API](https://platform.openai.com/) | `gpt-4o-mini` Automated lead priority analysis & summary |
| **Email Dispatch** | [Resend](https://resend.com/) | `6.17.2` Transactional notifications & welcome series |
| **Forms & Validation**| [React Hook Form](https://react-hook-form.com/) + [Zod](https://zod.dev/) | `7.x` / `4.x` Schema validation with custom error handling |
| **Icons** | [Lucide React](https://lucide.dev/) | `0.469.0` Sleek vector icons |

---

## 🚀 Getting Started Locally

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/webotixs-website.git
cd webotixs-website
npm install
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env.local` and populate your credentials:
```bash
cp .env.example .env.local
```

```env
# App Configuration
NEXT_PUBLIC_APP_URL=http://localhost:3000

# Supabase Project Credentials
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# OpenAI API Key for CRM Lead Classification
OPENAI_API_KEY=sk-your-openai-api-key

# Resend API Key for Email Notifications
RESEND_API_KEY=re_your_resend_api_key
NOTIFICATION_EMAIL=agency-leads@webotixs.com
```

### 3. Initialize Supabase Database Schema
1. Create a new project in your [Supabase Dashboard](https://supabase.com/dashboard).
2. Go to **SQL Editor** and open a new query block.
3. Copy the full contents of `supabase/migrations/20260716000000_initial_schema.sql` into the query window and click **Run**.
4. This script automatically:
   - Creates all 16 tables (`services`, `portfolio`, `team`, `blogs`, `industries`, `contact_submissions`, `newsletter_subscribers`, `careers`, `job_applications`, `media`, `activity_logs`, etc.).
   - Configures Row Level Security (RLS) policies for secure public submission (`contact` & `newsletter`) and authenticated admin CRUD.
   - Creates triggers and roles (`admin_users` table sync).

### 4. Create Your Admin User & Storage Bucket
1. In Supabase Dashboard, navigate to **Authentication → Users** and click **Add User** (or sign up via `signInWithPassword`).
2. Go to the **SQL Editor** and promote your email to an admin:
   ```sql
   INSERT INTO admin_users (id, email, full_name, role)
   VALUES ('your-auth-user-uuid', 'your@email.com', 'Admin User', 'super_admin')
   ON CONFLICT (email) DO UPDATE SET role = 'super_admin';
   ```
3. Navigate to **Storage → Buckets** and create a new public bucket named `media`.

### 5. Run the Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the public agency platform, and [http://localhost:3000/admin/login](http://localhost:3000/admin/login) to access the CMS dashboard.

---

## 📦 Production Build & Verification

To verify full static generation (`SSG`) and type checking across all 43+ routes:
```bash
# Run strict TypeScript check
npx tsc --noEmit

# Run Next.js Turbopack Production Build
npm run build
```

---

## 🌐 Deployment to Vercel

1. Push your repository to GitHub / GitLab / Bitbucket.
2. Import the project in your [Vercel Dashboard](https://vercel.com/new).
3. Under **Environment Variables**, add all keys from your `.env.local` (`NEXT_PUBLIC_SUPABASE_URL`, `OPENAI_API_KEY`, `RESEND_API_KEY`, etc.).
4. Click **Deploy**. Vercel will automatically detect Next.js 16 and optimize static and dynamic route handlers (`/api/contact`, `/api/newsletter`).

---

## 📁 Project Directory Structure

```
webotixs-website/
├── src/
│   ├── app/
│   │   ├── (public)/              # Public website routes (Home, About, Services, Portfolio, Blog, Contact, Careers, etc.)
│   │   ├── admin/
│   │   │   ├── login/             # Admin authentication login (`/admin/login`)
│   │   │   └── (dashboard)/       # Admin CMS modules (Services, Portfolio, Blog with Tiptap, Media, Careers, CRM, Logs, Backup)
│   │   ├── api/                   # Backend endpoints (`/api/contact` with OpenAI classification, `/api/newsletter`)
│   │   ├── layout.tsx             # Root dark-mode provider & font loading
│   │   └── not-found.tsx          # Custom 404 error page with animated compass
│   ├── components/
│   │   ├── admin/                 # RichTextEditor (Tiptap wrapper) and CMS UI tools
│   │   ├── animations/            # ParticleBackground, CustomCursor, ScrollReveal, AnimatedCounter, GradientMesh
│   │   └── public/                # Header (with i18n & RTL switcher), Footer, Homepage sections
│   ├── lib/
│   │   ├── data/                  # Mock data fallbacks for instant static rendering (`mock.ts`)
│   │   ├── supabase/              # Browser, server (`ssr`), and admin service clients
│   │   ├── types/                 # Comprehensive TypeScript interfaces (`types.ts`)
│   │   ├── utils/                 # Utility helpers (`cn`, `formatDate`, `slugify`, `calculateReadingTime`, etc.)
│   │   └── validations/           # Zod validation schemas (`contactSchema`, `newsletterSchema`)
│   └── proxy.ts                   # Next.js 16 proxy auth protection (replaces middleware)
├── supabase/
│   └── migrations/                # Complete SQL schema & RLS policies (`20260716000000_initial_schema.sql`)
├── public/                        # Static assets & icons
├── tailwind.config.ts             # Tailwind design tokens (`glass`, `#050816`, glowing borders)
├── globals.css                    # Tiptap prose styling, custom scrollbars, and background meshes
├── next.config.ts                 # Next.js optimization configuration
└── package.json                   # Dependencies and npm scripts
```

---

## 📜 License
© 2026 Webotixs Agency. All rights reserved. Designed and developed with cutting-edge engineering standards.
