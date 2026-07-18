-- =========================================================================
-- Webotixs Agency CRM & Operating System Schema (Scoped with crm_*)
-- Zero Breaking Changes to Existing Website & CMS Tables
-- =========================================================================

create extension if not exists "uuid-ossp";

-- 1. Roles Table (`crm_roles`)
create table if not exists public.crm_roles (
    id uuid default uuid_generate_v4() primary key,
    name varchar(100) unique not null,
    description text,
    level integer default 1,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 2. Permissions Table (`crm_permissions`)
create table if not exists public.crm_permissions (
    id uuid default uuid_generate_v4() primary key,
    name varchar(100) unique not null,
    description text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 3. CRM Users Table (`crm_users`)
create table if not exists public.crm_users (
    id uuid default uuid_generate_v4() primary key,
    auth_user_id uuid, -- Optional link to auth.users if logged via Supabase Auth
    email varchar(255) unique not null,
    full_name varchar(255) not null,
    avatar_url text,
    role_name varchar(100) not null references public.crm_roles(name) on update cascade,
    status varchar(20) default 'active' check (status in ('active', 'inactive')),
    phone varchar(50),
    department varchar(100),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 4. Clients Table (`crm_clients`)
create table if not exists public.crm_clients (
    id uuid default uuid_generate_v4() primary key,
    company_name varchar(255) not null,
    contact_name varchar(255) not null,
    email varchar(255) unique not null,
    phone varchar(50),
    website text,
    status varchar(20) default 'active' check (status in ('active', 'inactive', 'prospect')),
    notes text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 5. Client Credentials Table (`crm_client_credentials`)
create table if not exists public.crm_client_credentials (
    id uuid default uuid_generate_v4() primary key,
    client_id uuid references public.crm_clients(id) on delete cascade unique not null,
    portal_username varchar(255) not null,
    temp_password text not null,
    secret_token text not null,
    is_active boolean default true,
    last_login timestamp with time zone,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 6. Workflow Templates Table (`crm_workflow_templates`)
create table if not exists public.crm_workflow_templates (
    id uuid default uuid_generate_v4() primary key,
    name varchar(255) unique not null,
    description text,
    category varchar(100),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 7. Workflow Steps Table (`crm_workflow_steps`)
create table if not exists public.crm_workflow_steps (
    id uuid default uuid_generate_v4() primary key,
    template_id uuid references public.crm_workflow_templates(id) on delete cascade not null,
    step_order integer not null,
    title varchar(255) not null,
    description text,
    default_role varchar(100) references public.crm_roles(name) on update cascade not null,
    required_deliverable varchar(100), -- e.g., 'Figma Link', 'GitHub Link', 'SEO Report', 'QA Report'
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique (template_id, step_order)
);

-- 8. Projects Table (`crm_projects`)
create table if not exists public.crm_projects (
    id uuid default uuid_generate_v4() primary key,
    title varchar(255) not null,
    client_id uuid references public.crm_clients(id) on delete cascade not null,
    package_type varchar(100),
    budget numeric(12, 2) default 0.00,
    deadline date not null,
    priority varchar(20) default 'medium' check (priority in ('low', 'medium', 'high', 'urgent')),
    status varchar(30) default 'In Progress' check (status in ('Planning', 'In Progress', 'Review', 'Completed', 'On Hold')),
    progress_percentage integer default 0 check (progress_percentage >= 0 and progress_percentage <= 100),
    project_manager_id uuid references public.crm_users(id) on delete set null,
    workflow_template_id uuid references public.crm_workflow_templates(id) on delete set null,
    notes text,
    requirements text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 9. Project Members Table (`crm_project_members`)
create table if not exists public.crm_project_members (
    id uuid default uuid_generate_v4() primary key,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    user_id uuid references public.crm_users(id) on delete cascade not null,
    assigned_role varchar(100) references public.crm_roles(name) on update cascade not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique (project_id, user_id)
);

-- 10. Tasks Table (`crm_tasks`)
create table if not exists public.crm_tasks (
    id uuid default uuid_generate_v4() primary key,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    step_order integer not null,
    title varchar(255) not null,
    description text,
    assigned_to uuid references public.crm_users(id) on delete set null,
    role_required varchar(100) references public.crm_roles(name) on update cascade,
    status varchar(30) default 'Todo' check (status in ('Todo', 'In Progress', 'Review', 'Completed', 'Locked')),
    due_date date,
    deliverable_type varchar(100),
    deliverable_url text,
    deliverable_notes text,
    completed_at timestamp with time zone,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 11. Task Files Table (`crm_task_files`)
create table if not exists public.crm_task_files (
    id uuid default uuid_generate_v4() primary key,
    task_id uuid references public.crm_tasks(id) on delete cascade not null,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    file_name varchar(255) not null,
    file_url text not null,
    file_size integer default 0,
    file_type varchar(100),
    uploaded_by uuid references public.crm_users(id) on delete set null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 12. Project Files Table (`crm_project_files`)
create table if not exists public.crm_project_files (
    id uuid default uuid_generate_v4() primary key,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    file_name varchar(255) not null,
    file_url text not null,
    folder_category varchar(50) default 'Documents' check (folder_category in ('Figma', 'Documents', 'Images', 'Videos', 'Contracts', 'Invoices', 'Reports')),
    file_size integer default 0,
    uploaded_by_name varchar(255) not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 13. Comments Table (`crm_comments`)
create table if not exists public.crm_comments (
    id uuid default uuid_generate_v4() primary key,
    task_id uuid references public.crm_tasks(id) on delete cascade,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    author_name varchar(255) not null,
    author_avatar text,
    author_role varchar(100),
    content text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 14. Messages / Chat Threads Table (`crm_messages`)
create table if not exists public.crm_messages (
    id uuid default uuid_generate_v4() primary key,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    sender_name varchar(255) not null,
    sender_role varchar(100) not null,
    sender_avatar text,
    content text not null,
    attachment_url text,
    attachment_name text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 15. Notifications Table (`crm_notifications`)
create table if not exists public.crm_notifications (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.crm_users(id) on delete cascade,
    client_id uuid references public.crm_clients(id) on delete cascade,
    title varchar(255) not null,
    message text not null,
    type varchar(50) default 'info' check (type in ('info', 'task', 'file', 'comment', 'deadline', 'delivery')),
    link_url text,
    is_read boolean default false,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 16. Invoices Table (`crm_invoices`)
create table if not exists public.crm_invoices (
    id uuid default uuid_generate_v4() primary key,
    invoice_number varchar(100) unique not null,
    project_id uuid references public.crm_projects(id) on delete set null,
    client_id uuid references public.crm_clients(id) on delete cascade not null,
    amount numeric(12, 2) not null,
    tax_amount numeric(12, 2) default 0.00,
    total_amount numeric(12, 2) not null,
    status varchar(20) default 'Pending' check (status in ('Paid', 'Pending', 'Overdue', 'Cancelled')),
    issue_date date not null,
    due_date date not null,
    pdf_url text,
    notes text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 17. Payments Table (`crm_payments`)
create table if not exists public.crm_payments (
    id uuid default uuid_generate_v4() primary key,
    invoice_id uuid references public.crm_invoices(id) on delete cascade not null,
    amount_paid numeric(12, 2) not null,
    payment_method varchar(100) default 'Bank Transfer',
    transaction_id varchar(255),
    payment_date timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 18. Activity Logs (`crm_activity_logs`)
create table if not exists public.crm_activity_logs (
    id uuid default uuid_generate_v4() primary key,
    project_id uuid references public.crm_projects(id) on delete cascade,
    actor_name varchar(255) not null,
    actor_role varchar(100) not null,
    action varchar(255) not null,
    details text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 19. Support Tickets Table (`crm_support_tickets`)
create table if not exists public.crm_support_tickets (
    id uuid default uuid_generate_v4() primary key,
    ticket_number varchar(100) unique not null,
    client_id uuid references public.crm_clients(id) on delete cascade not null,
    project_id uuid references public.crm_projects(id) on delete set null,
    subject varchar(255) not null,
    description text not null,
    priority varchar(20) default 'medium' check (priority in ('low', 'medium', 'high', 'urgent')),
    status varchar(20) default 'Open' check (status in ('Open', 'Working', 'Solved', 'Closed')),
    assigned_to uuid references public.crm_users(id) on delete set null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 20. Project Status History Table (`crm_project_status_history`)
create table if not exists public.crm_project_status_history (
    id uuid default uuid_generate_v4() primary key,
    project_id uuid references public.crm_projects(id) on delete cascade not null,
    old_status varchar(50),
    new_status varchar(50) not null,
    changed_by_name varchar(255) not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable RLS on all CRM tables
alter table public.crm_roles enable row level security;
alter table public.crm_permissions enable row level security;
alter table public.crm_users enable row level security;
alter table public.crm_clients enable row level security;
alter table public.crm_client_credentials enable row level security;
alter table public.crm_workflow_templates enable row level security;
alter table public.crm_workflow_steps enable row level security;
alter table public.crm_projects enable row level security;
alter table public.crm_project_members enable row level security;
alter table public.crm_tasks enable row level security;
alter table public.crm_task_files enable row level security;
alter table public.crm_project_files enable row level security;
alter table public.crm_comments enable row level security;
alter table public.crm_messages enable row level security;
alter table public.crm_notifications enable row level security;
alter table public.crm_invoices enable row level security;
alter table public.crm_payments enable row level security;
alter table public.crm_activity_logs enable row level security;
alter table public.crm_support_tickets enable row level security;
alter table public.crm_project_status_history enable row level security;

-- Open RLS policies for CRM access (to allow admin/proxy & API actions freely)
create policy "Allow all operations on crm_roles" on public.crm_roles for all using (true) with check (true);
create policy "Allow all operations on crm_permissions" on public.crm_permissions for all using (true) with check (true);
create policy "Allow all operations on crm_users" on public.crm_users for all using (true) with check (true);
create policy "Allow all operations on crm_clients" on public.crm_clients for all using (true) with check (true);
create policy "Allow all operations on crm_client_credentials" on public.crm_client_credentials for all using (true) with check (true);
create policy "Allow all operations on crm_workflow_templates" on public.crm_workflow_templates for all using (true) with check (true);
create policy "Allow all operations on crm_workflow_steps" on public.crm_workflow_steps for all using (true) with check (true);
create policy "Allow all operations on crm_projects" on public.crm_projects for all using (true) with check (true);
create policy "Allow all operations on crm_project_members" on public.crm_project_members for all using (true) with check (true);
create policy "Allow all operations on crm_tasks" on public.crm_tasks for all using (true) with check (true);
create policy "Allow all operations on crm_task_files" on public.crm_task_files for all using (true) with check (true);
create policy "Allow all operations on crm_project_files" on public.crm_project_files for all using (true) with check (true);
create policy "Allow all operations on crm_comments" on public.crm_comments for all using (true) with check (true);
create policy "Allow all operations on crm_messages" on public.crm_messages for all using (true) with check (true);
create policy "Allow all operations on crm_notifications" on public.crm_notifications for all using (true) with check (true);
create policy "Allow all operations on crm_invoices" on public.crm_invoices for all using (true) with check (true);
create policy "Allow all operations on crm_payments" on public.crm_payments for all using (true) with check (true);
create policy "Allow all operations on crm_activity_logs" on public.crm_activity_logs for all using (true) with check (true);
create policy "Allow all operations on crm_support_tickets" on public.crm_support_tickets for all using (true) with check (true);
create policy "Allow all operations on crm_project_status_history" on public.crm_project_status_history for all using (true) with check (true);

-- Indexes for performance
create index if not exists idx_crm_projects_client on public.crm_projects(client_id);
create index if not exists idx_crm_tasks_project on public.crm_tasks(project_id);
create index if not exists idx_crm_tasks_assigned on public.crm_tasks(assigned_to);
create index if not exists idx_crm_invoices_client on public.crm_invoices(client_id);
create index if not exists idx_crm_tickets_client on public.crm_support_tickets(client_id);
create index if not exists idx_crm_messages_project on public.crm_messages(project_id);
create index if not exists idx_crm_activity_project on public.crm_activity_logs(project_id);

-- =========================================================================
-- Seed Data: 11 Roles & 7 Workflow Templates with Steps
-- =========================================================================

insert into public.crm_roles (id, name, description, level) values
  ('11111111-1111-1111-1111-111111111101', 'Super Admin', 'Full system access, role management, payments, and agency settings', 10),
  ('11111111-1111-1111-1111-111111111102', 'Admin', 'Creates projects, manages clients, invoices, and workflow templates', 9),
  ('11111111-1111-1111-1111-111111111103', 'Project Manager', 'Oversees project timelines, assigns team tasks, reviews deliverables', 8),
  ('11111111-1111-1111-1111-111111111104', 'UI/UX Designer', 'Responsible for wireframes, Figma prototypes, and visual identity', 5),
  ('11111111-1111-1111-1111-111111111105', 'WordPress Developer', 'Develops custom WordPress themes, plugins, and CMS integrations', 5),
  ('11111111-1111-1111-1111-111111111106', 'Frontend Developer', 'Builds responsive React/Next.js/Tailwind UI and micro-animations', 5),
  ('11111111-1111-1111-1111-111111111107', 'Backend Developer', 'Architects Supabase/PostgreSQL databases, APIs, and cloud services', 5),
  ('11111111-1111-1111-1111-111111111108', 'SEO Specialist', 'Conducts keyword research, technical SEO audits, and optimization reports', 5),
  ('11111111-1111-1111-1111-111111111109', 'Content Writer', 'Crafts conversion-focused copy, brand storytelling, and documentation', 5),
  ('11111111-1111-1111-1111-111111111110', 'QA Tester', 'Executes responsiveness checks, cross-browser testing, and bug reporting', 5),
  ('11111111-1111-1111-1111-111111111111', 'Client', 'External client with access to progress portal, files, invoices, and support', 1)
on conflict (name) do nothing;

-- Seed Team Users (`crm_users`)
insert into public.crm_users (id, email, full_name, role_name, status, phone, department) values
  ('22222222-2222-2222-2222-222222222201', 'ahmed@webotixs.com', 'Ahmed Al-Rashid', 'Super Admin', 'active', '+971 50 111 2233', 'Executive'),
  ('22222222-2222-2222-2222-222222222202', 'elena@webotixs.com', 'Elena Rostova', 'Project Manager', 'active', '+1 415 889 1020', 'Management'),
  ('22222222-2222-2222-2222-222222222203', 'sarah@webotixs.com', 'Sarah Chen', 'UI/UX Designer', 'active', '+1 650 443 9090', 'Design'),
  ('22222222-2222-2222-2222-222222222204', 'marcus@webotixs.com', 'Marcus Johnson', 'WordPress Developer', 'active', '+44 20 7946 0192', 'Engineering'),
  ('22222222-2222-2222-2222-222222222205', 'zayn@webotixs.com', 'Zayn Malik', 'Frontend Developer', 'active', '+971 52 443 8989', 'Engineering'),
  ('22222222-2222-2222-2222-222222222206', 'priya@webotixs.com', 'Priya Sharma', 'SEO Specialist', 'active', '+91 98 2001 3344', 'Marketing'),
  ('22222222-2222-2222-2222-222222222207', 'david@webotixs.com', 'David Kim', 'QA Tester', 'active', '+1 310 554 8877', 'Quality Assurance')
on conflict (email) do nothing;

-- Seed Workflow Templates (`crm_workflow_templates`)
insert into public.crm_workflow_templates (id, name, description, category) values
  ('33333333-3333-3333-3333-333333333301', 'WordPress Website', 'Complete agency lifecycle from UI Design to Development, SEO, QA, and Delivery', 'Web Development'),
  ('33333333-3333-3333-3333-333333333302', 'Shopify Store', 'High-converting e-commerce build including headless design, custom checkout, and product setup', 'E-Commerce'),
  ('33333333-3333-3333-3333-333333333303', 'Webflow Website', 'Modern enterprise Webflow build with custom GSAP/Framer animations and CMS schema', 'Web Development'),
  ('33333333-3333-3333-3333-333333333304', 'UI/UX Design Only', 'Comprehensive UX research, wireframing, high-fidelity UI design, and clickable prototype', 'Design & Branding'),
  ('33333333-3333-3333-3333-333333333305', 'SEO Project', 'In-depth technical SEO audit, on-page optimization, backlink strategy, and analytics setup', 'Digital Marketing'),
  ('33333333-3333-3333-3333-333333333306', 'AI Automation Project', 'Enterprise AI workflow mapping, custom LLM prompt engineering, and API integration', 'AI & Cloud'),
  ('33333333-3333-3333-3333-333333333307', 'Branding & Graphic Design', 'Full visual identity kit: brand discovery, logo design, typography, and PDF guidelines', 'Design & Branding')
on conflict (name) do nothing;

-- Seed Workflow Steps (`crm_workflow_steps`)
-- 1. WordPress Website Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333301', 1, 'UI/UX Design', 'Create wireframes and high-fidelity prototype in Figma', 'UI/UX Designer', 'Figma Link'),
  ('33333333-3333-3333-3333-333333333301', 2, 'Development', 'Build responsive custom theme and configure core plugins', 'WordPress Developer', 'Staging URL'),
  ('33333333-3333-3333-3333-333333333301', 3, 'SEO Optimization', 'Configure meta tags, sitemaps, schema markup, and speed metrics', 'SEO Specialist', 'SEO Report'),
  ('33333333-3333-3333-3333-333333333301', 4, 'QA Testing', 'Cross-browser testing, mobile responsiveness, forms, and bug fixes', 'QA Tester', 'QA Report'),
  ('33333333-3333-3333-3333-333333333301', 5, 'Delivery & Client Handoff', 'Final client presentation, domain launch, and documentation handoff', 'Project Manager', 'Sign-off Document');

-- 2. Shopify Store Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333302', 1, 'Store Strategy & Wireframes', 'Map e-commerce customer journey and cart conversion funnels', 'UI/UX Designer', 'Figma Link'),
  ('33333333-3333-3333-3333-333333333302', 2, 'Shopify Custom Development', 'Develop custom Liquid/Headless storefront and integrate payment gateways', 'Frontend Developer', 'Staging URL'),
  ('33333333-3333-3333-3333-333333333302', 3, 'Product Import & SEO', 'Configure product variants, collections, and e-commerce SEO metadata', 'SEO Specialist', 'SEO Report'),
  ('33333333-3333-3333-3333-333333333302', 4, 'QA & Checkout Testing', 'Perform end-to-end order simulation, shipping calculation, and speed audits', 'QA Tester', 'QA Report'),
  ('33333333-3333-3333-3333-333333333302', 5, 'Launch & Store Delivery', 'Go-live check, DNS connection, and client store training session', 'Project Manager', 'Launch Report');

-- 3. Webflow Website Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333303', 1, 'UI/UX & Animation Mapping', 'Design interactive components and map Framer/GSAP animations in Figma', 'UI/UX Designer', 'Figma Link'),
  ('33333333-3333-3333-3333-333333333303', 2, 'Webflow Build & CMS Setup', 'Build responsive client-first Webflow structure and dynamic CMS collections', 'Frontend Developer', 'Preview URL'),
  ('33333333-3333-3333-3333-333333333303', 3, 'Technical SEO Audit', 'Set up clean slugs, 301 redirects, Open Graph cards, and image compression', 'SEO Specialist', 'SEO Report'),
  ('33333333-3333-3333-3333-333333333303', 4, 'QA & Responsiveness Check', 'Test breakpoints across tablet, mobile landscape/portrait, and 4K displays', 'QA Tester', 'QA Report'),
  ('33333333-3333-3333-3333-333333333303', 5, 'Client Transfer & Delivery', 'Transfer Webflow site to client workspace and provide video instructions', 'Project Manager', 'Sign-off Document');

-- 4. UI/UX Design Only Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333304', 1, 'UX Research & Persona Mapping', 'Analyze target demographic and establish user flows and architecture', 'UI/UX Designer', 'UX Brief PDF'),
  ('33333333-3333-3333-3333-333333333304', 2, 'Low-Fidelity Wireframing', 'Create structural wireframes for all desktop and mobile pages', 'UI/UX Designer', 'Wireframes Link'),
  ('33333333-3333-3333-3333-333333333304', 3, 'High-Fidelity UI Design', 'Design pixel-perfect screens applying brand typography, color palette, and icons', 'UI/UX Designer', 'Figma Link'),
  ('33333333-3333-3333-3333-333333333304', 4, 'Interactive Prototyping', 'Connect screens with transitions and micro-interactions for user testing', 'UI/UX Designer', 'Prototype Link'),
  ('33333333-3333-3333-3333-333333333304', 5, 'Design System Handoff', 'Export design tokens, icon packs, and developer inspection documentation', 'Project Manager', 'Design Kit');

-- 5. SEO Project Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333305', 1, 'Comprehensive Technical Audit', 'Analyze crawlability, indexation, broken links, and Core Web Vitals', 'SEO Specialist', 'Audit Report PDF'),
  ('33333333-3333-3333-3333-333333333305', 2, 'Keyword & Competitor Matrix', 'Identify high-intent buyer keywords and map competitor content gaps', 'SEO Specialist', 'Keyword Matrix'),
  ('33333333-3333-3333-3333-333333333305', 3, 'On-Page Optimization', 'Optimize title tags, meta descriptions, H1 hierarchy, and internal linking', 'SEO Specialist', 'Optimization Logs'),
  ('33333333-3333-3333-3333-333333333305', 4, 'Content Strategy Execution', 'Produce optimized blog articles and landing page copy targets', 'Content Writer', 'Content Drafts'),
  ('33333333-3333-3333-3333-333333333305', 5, 'Analytics Dashboard & Reporting', 'Configure Google Search Console, GA4 goals, and executive monthly report', 'Project Manager', 'Monthly Report');

-- 6. AI Automation Project Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333306', 1, 'Workflow Mapping & Scoping', 'Map business processes and identify bottleneck steps ideal for AI automation', 'Project Manager', 'Architecture Diagram'),
  ('33333333-3333-3333-3333-333333333306', 2, 'AI Model Prompt Engineering', 'Design and test custom instructions and system prompts using OpenAI/Claude APIs', 'Backend Developer', 'Prompt Spec Document'),
  ('33333333-3333-3333-3333-333333333306', 3, 'API & Pipeline Integration', 'Build Webhooks, Zapier/Make automations, and custom Node.js middleware', 'Backend Developer', 'GitHub Repo'),
  ('33333333-3333-3333-3333-333333333306', 4, 'Edge Case & Security QA', 'Simulate heavy API load, rate limit handling, and hallucination safeguards', 'QA Tester', 'QA Report'),
  ('33333333-3333-3333-3333-333333333306', 5, 'Deployment & Team Training', 'Deploy production endpoints and conduct live team walkthrough session', 'Project Manager', 'Training Video');

-- 7. Branding & Graphic Design Steps
insert into public.crm_workflow_steps (template_id, step_order, title, description, default_role, required_deliverable) values
  ('33333333-3333-3333-3333-333333333307', 1, 'Brand Discovery & Moodboards', 'Conduct stakeholder interview and present 3 conceptual visual directions', 'UI/UX Designer', 'Moodboard Deck'),
  ('33333333-3333-3333-3333-333333333307', 2, 'Logo Exploration & Concepts', 'Design primary wordmark, pictorial mark, and responsive logo variations', 'UI/UX Designer', 'Logo Concepts Deck'),
  ('33333333-3333-3333-3333-333333333307', 3, 'Typography & Color System', 'Select primary/secondary typefaces and create accessible HSL/HEX color codes', 'UI/UX Designer', 'Color System Spec'),
  ('33333333-3333-3333-3333-333333333307', 4, 'Comprehensive Brand Guidelines', 'Assemble full PDF brand book covering usage rules, spacing, and applications', 'UI/UX Designer', 'Brand Book PDF'),
  ('33333333-3333-3333-3333-333333333307', 5, 'Final Asset Kit Delivery', 'Export vector SVG, EPS, transparent PNG, and social media templates', 'Project Manager', 'Asset Kit ZIP');

-- Seed Sample Active Clients (`crm_clients` & `crm_client_credentials`)
insert into public.crm_clients (id, company_name, contact_name, email, phone, website, status, notes) values
  ('44444444-4444-4444-4444-444444444401', 'Al-Khaleej Retail Group', 'Tariq Al-Mansoor', 'tariq@alkhaleej.ae', '+971 50 889 1234', 'https://alkhaleej.ae', 'active', 'Enterprise retail client across UAE & Saudi Arabia.'),
  ('44444444-4444-4444-4444-444444444402', 'Finch Investments', 'Robert Finch', 'r.finch@finchinvest.com', '+1 555 987 6543', 'https://finchinvest.com', 'active', 'Private wealth management firm seeking premium corporate redesign.'),
  ('44444444-4444-4444-4444-444444444403', 'LuxBrand Paris', 'Sophie Laurent', 'sophie@luxbrand.fr', '+33 6 1234 5678', 'https://luxbrand.fr', 'active', 'High-end French fashion label expanding digital e-commerce presence.')
on conflict (email) do nothing;

insert into public.crm_client_credentials (client_id, portal_username, temp_password, secret_token, is_active) values
  ('44444444-4444-4444-4444-444444444401', 'tariq_alkhaleej', 'Webotixs!Khaleej2026', 'token_khaleej_9981a', true),
  ('44444444-4444-4444-4444-444444444402', 'robert_finch', 'Webotixs!Finch2026', 'token_finch_4432b', true),
  ('44444444-4444-4444-4444-444444444403', 'sophie_luxbrand', 'Webotixs!Lux2026', 'token_luxbrand_7719c', true)
on conflict (client_id) do nothing;

-- Seed Sample Projects (`crm_projects`)
insert into public.crm_projects (id, title, client_id, package_type, budget, deadline, priority, status, progress_percentage, project_manager_id, workflow_template_id, notes, requirements) values
  ('55555555-5555-5555-5555-555555555501', 'Al-Khaleej E-Commerce Headless Storefront', '44444444-4444-4444-4444-444444444401', 'Enterprise E-Commerce', 45000.00, '2026-09-15', 'high', 'In Progress', 60, '22222222-2222-2222-2222-222222222202', '33333333-3333-3333-3333-333333333302', 'Shopify Plus headless setup with custom multi-currency checkout.', 'Must support Arabic RTL natively and sub-second page load speeds.'),
  ('55555555-5555-5555-5555-555555555502', 'Finch Investments Corporate Portal', '44444444-4444-4444-4444-444444444402', 'Custom Web Design', 18500.00, '2026-08-30', 'medium', 'In Progress', 20, '22222222-2222-2222-2222-222222222202', '33333333-3333-3333-3333-333333333301', 'Sleek dark-mode glassmorphic website with interactive portfolio graphs.', 'Clean typography and high-security investor login area.'),
  ('55555555-5555-5555-5555-555555555503', 'LuxBrand Paris Digital Rebrand & UI/UX', '44444444-4444-4444-4444-444444444403', 'Brand Identity + UI/UX', 32000.00, '2026-08-10', 'high', 'Review', 80, '22222222-2222-2222-2222-222222222202', '33333333-3333-3333-3333-333333333304', 'Minimalist luxury aesthetic for autumn 2026 collection rollout.', 'Ultra-smooth micro-animations and luxury lookbook grid.')
on conflict (id) do nothing;

-- Seed Project Members (`crm_project_members`)
insert into public.crm_project_members (project_id, user_id, assigned_role) values
  ('55555555-5555-5555-5555-555555555501', '22222222-2222-2222-2222-222222222202', 'Project Manager'),
  ('55555555-5555-5555-5555-555555555501', '22222222-2222-2222-2222-222222222203', 'UI/UX Designer'),
  ('55555555-5555-5555-5555-555555555501', '22222222-2222-2222-2222-222222222205', 'Frontend Developer'),
  ('55555555-5555-5555-5555-555555555501', '22222222-2222-2222-2222-222222222206', 'SEO Specialist'),
  ('55555555-5555-5555-5555-555555555501', '22222222-2222-2222-2222-222222222207', 'QA Tester'),
  ('55555555-5555-5555-5555-555555555502', '22222222-2222-2222-2222-222222222202', 'Project Manager'),
  ('55555555-5555-5555-5555-555555555502', '22222222-2222-2222-2222-222222222203', 'UI/UX Designer'),
  ('55555555-5555-5555-5555-555555555502', '22222222-2222-2222-2222-222222222204', 'WordPress Developer'),
  ('55555555-5555-5555-5555-555555555503', '22222222-2222-2222-2222-222222222202', 'Project Manager'),
  ('55555555-5555-5555-5555-555555555503', '22222222-2222-2222-2222-222222222203', 'UI/UX Designer')
on conflict do nothing;

-- Seed Tasks (`crm_tasks`) for Al-Khaleej E-Commerce (Shopify Template)
insert into public.crm_tasks (id, project_id, step_order, title, description, assigned_to, role_required, status, due_date, deliverable_type, deliverable_url) values
  ('66666666-6666-6666-6666-666666666601', '55555555-5555-5555-5555-555555555501', 1, 'Store Strategy & Wireframes', 'Map e-commerce customer journey and cart conversion funnels', '22222222-2222-2222-2222-222222222203', 'UI/UX Designer', 'Completed', '2026-07-25', 'Figma Link', 'https://figma.com/file/alkhaleej-store-v2'),
  ('66666666-6666-6666-6666-666666666602', '55555555-5555-5555-5555-555555555501', 2, 'Shopify Custom Development', 'Develop custom Liquid/Headless storefront and integrate payment gateways', '22222222-2222-2222-2222-222222222205', 'Frontend Developer', 'Completed', '2026-08-10', 'Staging URL', 'https://staging.alkhaleej-store.vercel.app'),
  ('66666666-6666-6666-6666-666666666603', '55555555-5555-5555-5555-555555555501', 3, 'Product Import & SEO', 'Configure product variants, collections, and e-commerce SEO metadata', '22222222-2222-2222-2222-222222222206', 'SEO Specialist', 'In Progress', '2026-08-25', 'SEO Report', null),
  ('66666666-6666-6666-6666-666666666604', '55555555-5555-5555-5555-555555555501', 4, 'QA & Checkout Testing', 'Perform end-to-end order simulation, shipping calculation, and speed audits', '22222222-2222-2222-2222-222222222207', 'QA Tester', 'Locked', '2026-09-05', 'QA Report', null),
  ('66666666-6666-6666-6666-666666666605', '55555555-5555-5555-5555-555555555501', 5, 'Launch & Store Delivery', 'Go-live check, DNS connection, and client store training session', '22222222-2222-2222-2222-222222222202', 'Project Manager', 'Locked', '2026-09-15', 'Launch Report', null),
-- Finch Investments Tasks (WordPress Template)
  ('66666666-6666-6666-6666-666666666606', '55555555-5555-5555-5555-555555555502', 1, 'UI/UX Design', 'Create wireframes and high-fidelity prototype in Figma', '22222222-2222-2222-2222-222222222203', 'UI/UX Designer', 'Completed', '2026-07-28', 'Figma Link', 'https://figma.com/file/finch-invest-portal'),
  ('66666666-6666-6666-6666-666666666607', '55555555-5555-5555-5555-555555555502', 2, 'Development', 'Build responsive custom theme and configure core plugins', '22222222-2222-2222-2222-222222222204', 'WordPress Developer', 'In Progress', '2026-08-15', 'Staging URL', null),
  ('66666666-6666-6666-6666-666666666608', '55555555-5555-5555-5555-555555555502', 3, 'SEO Optimization', 'Configure meta tags, sitemaps, schema markup, and speed metrics', '22222222-2222-2222-2222-222222222206', 'SEO Specialist', 'Locked', '2026-08-22', 'SEO Report', null),
  ('66666666-6666-6666-6666-666666666609', '55555555-5555-5555-5555-555555555502', 4, 'QA Testing', 'Cross-browser testing, mobile responsiveness, forms, and bug fixes', '22222222-2222-2222-2222-222222222207', 'QA Tester', 'Locked', '2026-08-28', 'QA Report', null),
  ('66666666-6666-6666-6666-666666666610', '55555555-5555-5555-5555-555555555502', 5, 'Delivery & Client Handoff', 'Final client presentation, domain launch, and documentation handoff', '22222222-2222-2222-2222-222222222202', 'Project Manager', 'Locked', '2026-08-30', 'Sign-off Document', null)
on conflict (id) do nothing;

-- Seed Invoices (`crm_invoices`)
insert into public.crm_invoices (id, invoice_number, project_id, client_id, amount, tax_amount, total_amount, status, issue_date, due_date, notes) values
  ('77777777-7777-7777-7777-777777777701', 'INV-2026-001', '55555555-5555-5555-5555-555555555501', '44444444-4444-4444-4444-444444444401', 22500.00, 1125.00, 23625.00, 'Paid', '2026-07-01', '2026-07-15', '50% Upfront Deposit for Al-Khaleej E-Commerce Overhaul.'),
  ('77777777-7777-7777-7777-777777777702', 'INV-2026-002', '55555555-5555-5555-5555-555555555502', '44444444-4444-4444-4444-444444444402', 9250.00, 0.00, 9250.00, 'Paid', '2026-07-10', '2026-07-24', '50% Milestone Deposit for Finch Investments Portal.'),
  ('77777777-7777-7777-7777-777777777703', 'INV-2026-003', '55555555-5555-5555-5555-555555555503', '44444444-4444-4444-4444-444444444403', 16000.00, 800.00, 16800.00, 'Pending', '2026-07-16', '2026-07-30', '50% Milestone Payment for LuxBrand Paris Rebrand.')
on conflict (invoice_number) do nothing;

-- Seed Support Tickets (`crm_support_tickets`)
insert into public.crm_support_tickets (id, ticket_number, client_id, project_id, subject, description, priority, status, assigned_to) values
  ('88888888-8888-8888-8888-888888888801', 'TICK-101', '44444444-4444-4444-4444-444444444401', '55555555-5555-5555-5555-555555555501', 'Staging Currency Dropdown Inquiry', 'Can we ensure the currency switcher on staging remembers user selection via cookie across tabs?', 'medium', 'Working', '2026-07-18'),
  ('88888888-8888-8888-8888-888888888802', 'TICK-102', '44444444-4444-4444-4444-444444444402', '55555555-5555-5555-5555-555555555502', 'Additional Team Member Access', 'We need to add our Chief Financial Officer to the client portal to review invoices.', 'low', 'Solved', '2026-07-17')
on conflict (ticket_number) do nothing;

-- Seed Activity Logs (`crm_activity_logs`)
insert into public.crm_activity_logs (project_id, actor_name, actor_role, action, details) values
  ('55555555-5555-5555-5555-555555555501', 'Ahmed Al-Rashid', 'Super Admin', 'Project Created', 'Created project Al-Khaleej E-Commerce Headless Storefront and auto-generated 5 workflow tasks.'),
  ('55555555-5555-5555-5555-555555555501', 'Sarah Chen', 'UI/UX Designer', 'Task Completed', 'Uploaded Figma link and marked Store Strategy & Wireframes as Completed. Unlocked Shopify Custom Development.'),
  ('55555555-5555-5555-5555-555555555501', 'Zayn Malik', 'Frontend Developer', 'Task Completed', 'Deployed staging build to Vercel and marked Shopify Custom Development as Completed. Unlocked Product Import & SEO.')
on conflict do nothing;
