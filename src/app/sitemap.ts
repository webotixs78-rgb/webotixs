import { MetadataRoute } from 'next'
import { mockServices, mockPortfolio, mockBlogPosts } from '@/lib/data/mock'
import { slugify } from '@/lib/utils'

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = 'https://www.webotixs.com'

  // Core static pages
  const staticRoutes = [
    '',
    '/services',
    '/portfolio',
    '/about',
    '/contact',
    '/careers',
    '/blog',
    '/industries',
    '/privacy-policy',
    '/terms',
    '/cookie-policy',
  ].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString(),
    changeFrequency: 'daily' as const,
    priority: route === '' ? 1.0 : 0.8,
  }))

  // Local SEO landing pages
  const locationRoutes = [
    '/dallas-tx',
    '/las-vegas-nv',
    '/boston-ma',
    '/leeds-uk',
    '/phoenix-az',
    '/locations/dallas-tx',
    '/locations/las-vegas-nv',
    '/locations/boston-ma',
    '/locations/leeds-uk',
    '/locations/phoenix-az',
  ].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString(),
    changeFrequency: 'weekly' as const,
    priority: 0.8,
  }))

  // Dynamic Service routes
  const serviceRoutes = mockServices.map((service) => ({
    url: `${baseUrl}/services/${service.slug}`,
    lastModified: service.updated_at || new Date().toISOString(),
    changeFrequency: 'weekly' as const,
    priority: 0.7,
  }))

  // Dynamic Portfolio routes
  const portfolioRoutes = mockPortfolio.map((project: any) => ({
    url: `${baseUrl}/portfolio/${project.slug || slugify(project.title) || project.id}`,
    lastModified: project.updated_at || new Date().toISOString(),
    changeFrequency: 'weekly' as const,
    priority: 0.7,
  }))

  // Dynamic Blog routes
  const blogRoutes = mockBlogPosts.map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: post.updated_at || new Date().toISOString(),
    changeFrequency: 'weekly' as const,
    priority: 0.6,
  }))

  return [...staticRoutes, ...locationRoutes, ...serviceRoutes, ...portfolioRoutes, ...blogRoutes]
}
