import { z } from 'zod'

export const contactSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters').max(100),
  email: z.string().email('Please enter a valid email address'),
  phone: z.string().optional(),
  company: z.string().optional(),
  service: z.string().optional(),
  budget: z.string().optional(),
  message: z.string().min(20, 'Message must be at least 20 characters').max(2000),
})

export type ContactFormData = z.infer<typeof contactSchema>

export const newsletterSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  name: z.string().optional(),
})

export type NewsletterFormData = z.infer<typeof newsletterSchema>
