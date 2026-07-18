// =====================
// Database Types
// =====================

export interface Service {
  id: string
  title: string
  slug: string
  icon: string
  cover_image: string | null
  short_description: string
  long_description: string
  features: string[]
  process: ProcessStep[]
  gallery: string[]
  seo: SEOMeta
  status: 'published' | 'draft'
  featured: boolean
  order: number
  created_at: string
  updated_at: string
}

export interface ProcessStep {
  step: number
  title: string
  description: string
}

export interface PortfolioProject {
  id: string
  title: string
  client: string
  industry: string
  thumbnail: string | null
  gallery: string[]
  description: string
  challenge: string
  solution: string
  results: string[]
  technologies: string[]
  live_url: string | null
  featured: boolean
  order: number
  seo: SEOMeta
  status: 'published' | 'draft'
  created_at: string
  updated_at: string
}

export interface TeamMember {
  id: string
  name: string
  position: string
  photo: string | null
  bio: string
  linkedin: string | null
  github: string | null
  email: string | null
  order: number
  featured: boolean
  status: 'active' | 'inactive'
  portal_password?: string
  role_department?: string
  created_at: string
}

export interface BlogPost {
  id: string
  title: string
  slug: string
  excerpt: string
  content: string
  featured_image: string | null
  gallery: string[]
  author_id: string
  author?: TeamMember
  category_id: string
  category?: BlogCategory
  tags: string[]
  seo: SEOMeta
  status: 'published' | 'draft'
  published_at: string | null
  reading_time: number
  related_posts: string[]
  created_at: string
  updated_at: string
}

export interface BlogCategory {
  id: string
  name: string
  slug: string
  description: string | null
  created_at: string
}

export interface Industry {
  id: string
  title: string
  slug: string
  image: string | null
  description: string
  benefits: string[]
  cta: IndustryCTA
  seo: SEOMeta
  status: 'published' | 'draft'
  order: number
  created_at: string
}

export interface IndustryCTA {
  title: string
  description: string
  button_text: string
  button_url: string
}

export interface Testimonial {
  id: string
  name: string
  position: string
  company: string
  avatar: string | null
  content: string
  rating: number
  featured: boolean
  order: number
  created_at: string
}

export interface Partner {
  id: string
  name: string
  logo: string
  url: string | null
  order: number
  created_at: string
}

export interface ContactSubmission {
  id: string
  name: string
  email: string
  phone: string | null
  company: string | null
  service: string | null
  budget: string | null
  message: string
  ai_priority: 'high' | 'medium' | 'low' | null
  ai_summary: string | null
  status: 'new' | 'contacted' | 'qualified' | 'closed'
  created_at: string
}

export interface NewsletterSubscriber {
  id: string
  email: string
  name: string | null
  status: 'active' | 'unsubscribed'
  created_at: string
}

export interface CareerJob {
  id: string
  title: string
  slug: string
  department: string
  location: string
  type: 'full-time' | 'part-time' | 'contract' | 'internship'
  description: string
  requirements: string[]
  benefits: string[]
  salary_range: string | null
  status: 'open' | 'closed'
  deadline: string | null
  created_at: string
}

export interface SiteSettings {
  id: string
  site_name: string
  logo_dark: string | null
  logo_light: string | null
  favicon: string | null
  primary_color: string
  secondary_color: string
  footer_text: string
  emails: string[]
  phone_numbers: string[]
  social_media: SocialMedia
  google_analytics_id: string | null
  smtp_config: SMTPConfig | null
  address: string | null
  google_map_url: string | null
  business_hours: BusinessHours | null
  whatsapp: string | null
  updated_at: string
}

export interface SocialMedia {
  facebook?: string
  twitter?: string
  instagram?: string
  linkedin?: string
  youtube?: string
  github?: string
  tiktok?: string
}

export interface SMTPConfig {
  host: string
  port: number
  user: string
  pass: string
  from_name: string
  from_email: string
}

export interface BusinessHours {
  monday?: string
  tuesday?: string
  wednesday?: string
  thursday?: string
  friday?: string
  saturday?: string
  sunday?: string
}

export interface SEOMeta {
  title?: string
  description?: string
  keywords?: string[]
  og_image?: string
  og_title?: string
  og_description?: string
  twitter_card?: string
  canonical?: string
  robots?: string
  schema?: object
}

// =====================
// Admin / Auth Types
// =====================

export type AdminRole = 'super_admin' | 'admin' | 'editor' | 'content_writer'

export interface AdminUser {
  id: string
  email: string
  full_name: string
  avatar: string | null
  role: AdminRole
  status: 'active' | 'inactive'
  created_at: string
  last_login: string | null
}

export interface ActivityLog {
  id: string
  user_id: string
  user?: AdminUser
  action: string
  entity_type: string
  entity_id: string | null
  details: Record<string, unknown>
  ip_address: string | null
  created_at: string
}

export interface MediaFile {
  id: string
  name: string
  original_name: string
  url: string
  type: 'image' | 'video' | 'svg' | 'pdf' | 'document'
  size: number
  bucket: string
  folder_id: string | null
  created_at: string
}

export interface MediaFolder {
  id: string
  name: string
  parent_id: string | null
  created_at: string
}

// =====================
// UI / Component Types
// =====================

export interface NavItem {
  label: string
  href: string
  children?: NavItem[]
}

export interface StatItem {
  value: string
  label: string
  suffix?: string
}

export interface Feature {
  icon: string
  title: string
  description: string
}

export interface PaginationInfo {
  page: number
  limit: number
  total: number
  totalPages: number
}

export interface ApiResponse<T> {
  data: T | null
  error: string | null
  success: boolean
}
