-- ==========================================
-- Webotixs Database Initialization Schema
-- ==========================================

-- Enable required extensions
create extension if not exists "uuid-ossp";

-- 1. Roles & Permissions Management
create table public.roles (
    id uuid default uuid_generate_v4() primary key,
    name varchar(50) unique not null,
    description text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.permissions (
    id uuid default uuid_generate_v4() primary key,
    name varchar(100) unique not null,
    description text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.role_permissions (
    role_id uuid references public.roles(id) on delete cascade,
    permission_id uuid references public.permissions(id) on delete cascade,
    primary key (role_id, permission_id)
);

-- 2. Admin Users Table (Linked to Supabase Auth)
create table public.admin_users (
    id uuid references auth.users(id) on delete cascade primary key,
    email varchar(255) unique not null,
    full_name varchar(255) not null,
    avatar_url text,
    role_id uuid references public.roles(id) on delete set null,
    status varchar(20) default 'active' check (status in ('active', 'inactive')),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    last_login timestamp with time zone
);

-- RLS (Row Level Security) for Admin Users
alter table public.admin_users enable row level security;

-- 3. Activity Logs for Admin Audits
create table public.activity_logs (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.admin_users(id) on delete set null,
    action varchar(255) not null,
    entity_type varchar(100) not null,
    entity_id uuid,
    details jsonb default '{}'::jsonb,
    ip_address varchar(45),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.activity_logs enable row level security;

-- 4. Site Settings (Global Configurations)
create table public.settings (
    id uuid default uuid_generate_v4() primary key,
    site_name varchar(100) default 'Webotixs' not null,
    logo_dark text,
    logo_light text,
    favicon text,
    primary_color varchar(20) default '#3B82F6',
    secondary_color varchar(20) default '#06B6D4',
    footer_text text default '© 2026 Webotixs. All rights reserved.',
    emails text[] default '{}'::text[],
    phone_numbers text[] default '{}'::text[],
    social_media jsonb default '{}'::jsonb,
    google_analytics_id varchar(50),
    smtp_config jsonb default '{}'::jsonb,
    address text,
    google_map_url text,
    business_hours jsonb default '{}'::jsonb,
    whatsapp varchar(50),
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 5. Services CMS
create table public.services (
    id uuid default uuid_generate_v4() primary key,
    title varchar(255) not null,
    slug varchar(255) unique not null,
    icon varchar(50) default 'Globe' not null,
    cover_image text,
    short_description text not null,
    long_description text not null,
    features text[] default '{}'::text[],
    process jsonb default '[]'::jsonb, -- array of {step: int, title: string, description: string}
    gallery text[] default '{}'::text[],
    seo jsonb default '{}'::jsonb,
    status varchar(20) default 'draft' check (status in ('published', 'draft')),
    featured boolean default false,
    "order" integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 6. Portfolio/Projects CMS
create table public.portfolio (
    id uuid default uuid_generate_v4() primary key,
    title varchar(255) not null,
    client varchar(255) not null,
    industry varchar(100) not null,
    thumbnail text,
    gallery text[] default '{}'::text[],
    description text not null,
    challenge text,
    solution text,
    results text[] default '{}'::text[],
    technologies text[] default '{}'::text[],
    live_url text,
    featured boolean default false,
    "order" integer default 0,
    seo jsonb default '{}'::jsonb,
    status varchar(20) default 'draft' check (status in ('published', 'draft')),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 7. Team Members CMS
create table public.team (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) not null,
    position varchar(255) not null,
    photo text,
    bio text,
    linkedin text,
    github text,
    email varchar(255),
    "order" integer default 0,
    featured boolean default false,
    status varchar(20) default 'active' check (status in ('active', 'inactive')),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 8. Blog Categories & Posts CMS
create table public.blog_categories (
    id uuid default uuid_generate_v4() primary key,
    name varchar(100) unique not null,
    slug varchar(100) unique not null,
    description text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.blogs (
    id uuid default uuid_generate_v4() primary key,
    title varchar(255) not null,
    slug varchar(255) unique not null,
    excerpt text not null,
    content text not null,
    featured_image text,
    gallery text[] default '{}'::text[],
    author_id uuid references public.team(id) on delete set null,
    category_id uuid references public.blog_categories(id) on delete set null,
    tags text[] default '{}'::text[],
    seo jsonb default '{}'::jsonb,
    status varchar(20) default 'draft' check (status in ('published', 'draft')),
    published_at timestamp with time zone,
    reading_time integer default 0,
    related_posts uuid[] default '{}'::uuid[],
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 9. Industries CMS
create table public.industries (
    id uuid default uuid_generate_v4() primary key,
    title varchar(255) not null,
    slug varchar(255) unique not null,
    image text,
    description text not null,
    benefits text[] default '{}'::text[],
    cta jsonb default '{}'::jsonb, -- {title, description, button_text, button_url}
    seo jsonb default '{}'::jsonb,
    status varchar(20) default 'draft' check (status in ('published', 'draft')),
    "order" integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 10. Partners & Clients (Logos)
create table public.partners (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) not null,
    logo text not null,
    url text,
    "order" integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 11. Testimonials
create table public.testimonials (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) not null,
    position varchar(255) not null,
    company varchar(255) not null,
    avatar text,
    content text not null,
    rating integer default 5 check (rating >= 1 and rating <= 5),
    featured boolean default false,
    "order" integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 12. CRM Leads & Form Submissions
create table public.contact_inquiries (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) not null,
    email varchar(255) not null,
    phone varchar(50),
    service varchar(100),
    message text not null,
    inquiry_source varchar(100) default 'Website Contact Form',
    status varchar(50) default 'New',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    ip_address varchar(100),
    browser text,
    country varchar(100)
);

alter table public.contact_inquiries enable row level security;

create table public.contact_submissions (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) not null,
    email varchar(255) not null,
    phone varchar(50),
    company varchar(255),
    service varchar(100),
    budget varchar(50),
    message text not null,
    ai_priority varchar(20) check (ai_priority in ('high', 'medium', 'low')),
    ai_summary text,
    status varchar(20) default 'new' check (status in ('new', 'contacted', 'qualified', 'closed')),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- RLS Configuration for Public Submissions (Public write, Admin read)
alter table public.contact_submissions enable row level security;

-- 13. Newsletter Subscribers
create table public.newsletter_subscribers (
    id uuid default uuid_generate_v4() primary key,
    email varchar(255) unique not null,
    name varchar(255),
    status varchar(20) default 'active' check (status in ('active', 'unsubscribed')),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.newsletter_subscribers enable row level security;

-- 14. Careers & Job Listings
create table public.careers (
    id uuid default uuid_generate_v4() primary key,
    title varchar(255) not null,
    slug varchar(255) unique not null,
    department varchar(100) not null,
    location varchar(100) not null,
    type varchar(50) check (type in ('full-time', 'part-time', 'contract', 'internship')),
    description text not null,
    requirements text[] default '{}'::text[],
    benefits text[] default '{}'::text[],
    salary_range varchar(100),
    status varchar(20) default 'open' check (status in ('open', 'closed')),
    deadline timestamp with time zone,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 15. Job Applications
create table public.job_applications (
    id uuid default uuid_generate_v4() primary key,
    career_id uuid references public.careers(id) on delete set null,
    name varchar(255) not null,
    email varchar(255) not null,
    phone varchar(50),
    resume_url text not null,
    cover_letter text,
    status varchar(20) default 'new' check (status in ('new', 'reviewing', 'interviewed', 'offered', 'rejected')),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.job_applications enable row level security;

-- 16. Media Library
create table public.media_folders (
    id uuid default uuid_generate_v4() primary key,
    name varchar(100) not null,
    parent_id uuid references public.media_folders(id) on delete cascade,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table public.media (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) not null,
    original_name varchar(255) not null,
    url text not null,
    type varchar(50) check (type in ('image', 'video', 'svg', 'pdf', 'document')),
    size integer not null,
    bucket varchar(50) not null,
    folder_id uuid references public.media_folders(id) on delete set null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.media enable row level security;
alter table public.media_folders enable row level security;


-- ==========================================
-- Triggers and Automatic Triggers Setup
-- ==========================================

-- Function to handle user signups automatically (synchronize Supabase auth to public.admin_users)
create or replace function public.handle_new_admin_user()
returns trigger as $$
begin
  insert into public.admin_users (id, email, full_name, role_id, status)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', 'Admin User'),
    (select id from public.roles where name = 'admin' limit 1),
    'active'
  );
  return new;
end;
$$ language plpgsql security definer;

-- Trigger to execute the above function on user signup
create or replace trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_admin_user();


-- ==========================================
-- Row Level Security (RLS) Policy Definitions
-- ==========================================

-- Admin Users Policies
create policy "Allow admins to read all admin_users"
    on public.admin_users for select
    using (auth.role() = 'authenticated');

create policy "Allow admins to modify admin_users"
    on public.admin_users for all
    using (auth.role() = 'authenticated');

-- Contact Submissions Policies
create policy "Allow public to insert contact_inquiries"
    on public.contact_inquiries for insert
    with check (true);

create policy "Allow authenticated to manage contact_inquiries"
    on public.contact_inquiries for all
    using (auth.role() = 'authenticated');

create policy "Allow public to insert contact_submissions"
    on public.contact_submissions for insert
    with check (true);

create policy "Allow authenticated to manage contact_submissions"
    on public.contact_submissions for all
    using (auth.role() = 'authenticated');

-- Newsletter Subscribers Policies
create policy "Allow public to insert newsletter_subscribers"
    on public.newsletter_subscribers for insert
    with check (true);

create policy "Allow authenticated to manage newsletter_subscribers"
    on public.newsletter_subscribers for all
    using (auth.role() = 'authenticated');

-- Job Applications Policies
create policy "Allow public to insert job_applications"
    on public.job_applications for insert
    with check (true);

create policy "Allow authenticated to manage job_applications"
    on public.job_applications for all
    using (auth.role() = 'authenticated');
