# enhanced_social_scraper.py

import asyncio
import aiohttp
import streamlit as st
import pandas as pd
import os
import json
import re
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Enhanced social media domains with platform identification
SOCIAL_DOMAINS = {
    "facebook.com": "Facebook",
    "instagram.com": "Instagram", 
    "twitter.com": "Twitter",
    "x.com": "Twitter",
    "linkedin.com": "LinkedIn",
    "youtube.com": "YouTube",
    "tiktok.com": "TikTok",
    "pinterest.com": "Pinterest",
    "snapchat.com": "Snapchat",
    "telegram.org": "Telegram",
    "telegram.me": "Telegram",
    "discord.gg": "Discord",
    "reddit.com": "Reddit"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Cache management
CACHE_FILE = "scraper_cache.json"

def load_cache():
    """Load cached results"""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache_data):
    """Save results to cache"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)

def get_cache_key(url):
    """Generate cache key for URL"""
    return hashlib.md5(url.encode()).hexdigest()

def validate_url(url):
    """Validate and normalize URLs"""
    if not url or not isinstance(url, str):
        return None
    
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
        if parsed.netloc and parsed.scheme:
            return url
    except:
        pass
    return None

def clean_social_url(url):
    """Clean and validate social media URLs"""
    if not url:
        return url
    
    # Remove tracking parameters
    url = re.sub(r'[?&]utm_[^&]*', '', url)
    url = re.sub(r'[?&]fbclid=[^&]*', '', url)
    url = re.sub(r'[?&]ref=[^&]*', '', url)
    
    # Remove trailing slashes and fragments
    url = re.sub(r'#.*$', '', url)
    url = url.rstrip('/')
    
    return url

def identify_platform(url):
    """Identify social media platform from URL"""
    if not url:
        return "Unknown"
    
    for domain, platform in SOCIAL_DOMAINS.items():
        if domain in url.lower():
            return platform
    return "Other"

def can_crawl(url):
    """Check if crawling is allowed by robots.txt"""
    try:
        rp = RobotFileParser()
        rp.set_url(urljoin(url, '/robots.txt'))
        rp.read()
        return rp.can_fetch(HEADERS['User-Agent'], url)
    except:
        return True  # Default to allow if can't read robots.txt

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_with_retry(session, url):
    """Fetch URL with retry logic"""
    try:
        async with session.get(url, headers=HEADERS, timeout=15) as response:
            if response.status == 200:
                return await response.text()
            else:
                logging.warning(f"HTTP {response.status} for {url}")
                return ""
    except asyncio.TimeoutError:
        logging.error(f"Timeout for {url}")
        return ""
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""

def extract_links_from_html(html, base_url):
    """Extract all links from HTML with improved parsing"""
    if not html:
        return set()
    
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    
    # Look for links in common social media locations
    selectors = [
        'a[href*="facebook"]',
        'a[href*="instagram"]', 
        'a[href*="twitter"]',
        'a[href*="linkedin"]',
        'a[href*="youtube"]',
        'a[href*="tiktok"]',
        'a[href*="pinterest"]',
        '.social-links a',
        '.footer a',
        '.contact a',
        'a[href]'  # Fallback to all links
    ]
    
    for selector in selectors:
        for a in soup.select(selector):
            href = a.get('href')
            if href:
                full_url = urljoin(base_url, href)
                links.add(full_url)
    
    return links

def filter_social_links(links, selected_platforms=None):
    """Filter and categorize social media links"""
    social_links = {}
    
    for link in links:
        link_lower = link.lower()
        for domain, platform in SOCIAL_DOMAINS.items():
            if domain in link_lower:
                if selected_platforms is None or platform in selected_platforms:
                    cleaned_url = clean_social_url(link)
                    if cleaned_url not in social_links:
                        social_links[cleaned_url] = {
                            'platform': platform,
                            'url': cleaned_url
                        }
    
    return social_links

async def crawl_site(session, base_url, max_pages=3, selected_platforms=None):
    """Crawl a single website with improved strategy"""
    cache = load_cache()
    cache_key = get_cache_key(base_url)
    
    # Check cache first
    if cache_key in cache:
        cached_time = cache[cache_key].get('timestamp', 0)
        if datetime.now().timestamp() - cached_time < 86400:  # 24 hours
            logging.info(f"Using cached result for {base_url}")
            return cache[cache_key]['result']
    
    # Check robots.txt
    if not can_crawl(base_url):
        logging.warning(f"Crawling not allowed for {base_url}")
        return create_empty_result(base_url)
    
    visited = set()
    social_links_found = {}
    
    # Priority pages to check
    priority_pages = [
        base_url,
        f"{base_url.rstrip('/')}/about",
        f"{base_url.rstrip('/')}/contact",
        f"{base_url.rstrip('/')}/team",
        f"{base_url.rstrip('/')}/footer"
    ]
    
    pages_checked = 0
    for page_url in priority_pages:
        if pages_checked >= max_pages:
            break
            
        if page_url in visited:
            continue
            
        visited.add(page_url)
        pages_checked += 1
        
        html = await fetch_with_retry(session, page_url)
        if not html:
            continue
        
        links = extract_links_from_html(html, base_url)
        new_social_links = filter_social_links(links, selected_platforms)
        
        for url, info in new_social_links.items():
            if url not in social_links_found:
                social_links_found[url] = {
                    'platform': info['platform'],
                    'found_on': page_url
                }
        
        # If we found social links, we can stop early
        if social_links_found:
            break
    
    # Create result
    result = create_result(base_url, social_links_found)
    
    # Cache the result
    cache[cache_key] = {
        'result': result,
        'timestamp': datetime.now().timestamp()
    }
    save_cache(cache)
    
    return result

def create_empty_result(base_url):
    """Create empty result structure"""
    return {
        "Website": base_url,
        "Social Links Found": 0,
        "Platforms": "",
        "Social Link 1": "",
        "Platform 1": "",
        "Found On Page 1": "",
        "Social Link 2": "",
        "Platform 2": "",
        "Found On Page 2": "",
        "Social Link 3": "",
        "Platform 3": "",
        "Found On Page 3": "",
        "Status": "No social links found"
    }

def create_result(base_url, social_links_found):
    """Create formatted result from social links"""
    result = {
        "Website": base_url,
        "Social Links Found": len(social_links_found),
        "Platforms": ", ".join(set(info['platform'] for info in social_links_found.values())),
        "Status": "Success" if social_links_found else "No social links found"
    }
    
    # Add up to 3 social links
    for i in range(1, 4):
        result[f"Social Link {i}"] = ""
        result[f"Platform {i}"] = ""
        result[f"Found On Page {i}"] = ""
    
    for i, (url, info) in enumerate(social_links_found.items(), 1):
        if i > 3:  # Limit to 3 results
            break
        result[f"Social Link {i}"] = url
        result[f"Platform {i}"] = info['platform']
        result[f"Found On Page {i}"] = info['found_on']
    
    return result

async def process_websites(websites, max_concurrent=5, max_pages=3, selected_platforms=None, progress_callback=None):
    """Process multiple websites with concurrent limiting"""
    semaphore = Semaphore(max_concurrent)
    results = []
    
    async def bounded_crawl(session, url, index):
        async with semaphore:
            result = await crawl_site(session, url, max_pages, selected_platforms)
            if progress_callback:
                progress_callback(index + 1, len(websites), url)
            return result
    
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max_concurrent),
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        tasks = [bounded_crawl(session, url, i) for i, url in enumerate(websites)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(f"Error processing {websites[i]}: {result}")
            valid_results.append(create_empty_result(websites[i]))
        else:
            valid_results.append(result)
    
    return valid_results

def export_results(results, output_format="both"):
    """Export results with enhanced formatting"""
    df = pd.DataFrame(results)
    
    # Create summary statistics
    total_sites = len(df)
    sites_with_social = len(df[df['Social Links Found'] > 0])
    success_rate = (sites_with_social / total_sites * 100) if total_sites > 0 else 0
    
    # Platform statistics
    platform_counts = {}
    for _, row in df.iterrows():
        platforms = row['Platforms'].split(', ') if row['Platforms'] else []
        for platform in platforms:
            if platform.strip():
                platform_counts[platform.strip()] = platform_counts.get(platform.strip(), 0) + 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files_created = []
    
    if output_format in ["csv", "both"]:
        csv_path = f"social_links_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        files_created.append(csv_path)
    
    if output_format in ["excel", "both"]:
        xlsx_path = f"social_links_{timestamp}.xlsx"
        
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name='Social Links', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Sites Processed', 'Sites with Social Links', 'Success Rate (%)', 'Most Common Platform'],
                'Value': [
                    total_sites,
                    sites_with_social,
                    f"{success_rate:.1f}%",
                    max(platform_counts.items(), key=lambda x: x[1])[0] if platform_counts else 'None'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Platform statistics
            if platform_counts:
                platform_df = pd.DataFrame(list(platform_counts.items()), columns=['Platform', 'Count'])
                platform_df = platform_df.sort_values('Count', ascending=False)
                platform_df.to_excel(writer, sheet_name='Platform Stats', index=False)
        
        files_created.append(xlsx_path)
    
    return files_created

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Enhanced Social Media Link Scraper",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Enhanced Social Media Link Scraper")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    max_pages = st.sidebar.slider("Max pages per site", 1, 10, 3, 
                                 help="Maximum number of pages to crawl per website")
    
    max_concurrent = st.sidebar.slider("Concurrent requests", 1, 20, 5,
                                      help="Number of websites to process simultaneously")
    
    selected_platforms = st.sidebar.multiselect(
        "Social platforms to find:",
        list(SOCIAL_DOMAINS.values()),
        default=list(SOCIAL_DOMAINS.values())[:6],  # Default to first 6 platforms
        help="Select which social media platforms to search for"
    )
    
    export_format = st.sidebar.selectbox(
        "Export format:",
        ["both", "csv", "excel"],
        help="Choose output file format"
    )
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear Cache"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.sidebar.success("Cache cleared!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Instructions:
        1. Upload a CSV file with a **`Website`** column
        2. Configure settings in the sidebar
        3. Click "Start Scraping" to begin
        
        **Features:**
        - ✅ Finds social media links from multiple platforms
        - ✅ Respects robots.txt and implements rate limiting
        - ✅ Caches results to avoid re-processing
        - ✅ Exports to CSV and Excel with statistics
        - ✅ Real-time progress tracking
        """)
    
    with col2:
        # Sample data
        sample_df = pd.DataFrame({
            "Website": [
                "https://example.com",
                "https://github.com",
                "https://stackoverflow.com"
            ]
        })
        
        st.download_button(
            "⬇️ Download Sample CSV",
            sample_df.to_csv(index=False),
            file_name="sample_websites.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to continue.")
        return
    
    # Process uploaded file
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        return
    
    if "Website" not in df_input.columns:
        st.error("❌ CSV must contain a 'Website' column.")
        return
    
    # Validate and clean URLs
    raw_websites = df_input["Website"].dropna().tolist()
    validated_websites = [validate_url(url) for url in raw_websites]
    websites = [url for url in validated_websites if url is not None]
    
    invalid_count = len(raw_websites) - len(websites)
    
    if invalid_count > 0:
        st.warning(f"⚠️ {invalid_count} invalid URLs were skipped.")
    
    if not websites:
        st.error("❌ No valid websites found in the uploaded file.")
        return
    
    # Display preview
    st.markdown("### 📊 Preview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Valid URLs", len(websites))
    with col2:
        st.metric("Platforms to Search", len(selected_platforms))
    with col3:
        st.metric("Max Pages per Site", max_pages)
    
    # Show first few URLs
    with st.expander("📝 URLs to Process"):
        st.write(websites[:10])
        if len(websites) > 10:
            st.write(f"... and {len(websites) - 10} more")
    
    # Start scraping
    if st.button("🚀 Start Scraping", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        results_placeholder = st.empty()
        
        def update_progress(current, total, current_url):
            progress = current / total
            progress_bar.progress(progress)
            progress_text.text(f"Processing {current}/{total}: {current_url}")
        
        start_time = datetime.now()
        
        with st.spinner("Scraping websites... Please wait."):
            try:
                results = asyncio.run(process_websites(
                    websites,
                    max_concurrent=max_concurrent,
                    max_pages=max_pages,
                    selected_platforms=selected_platforms,
                    progress_callback=update_progress
                ))
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                progress_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Scraping completed in {duration:.1f} seconds!")
                
                # Display results
                df_results = pd.DataFrame(results)
                
                # Summary statistics
                st.markdown("### 📈 Results Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                sites_with_social = len(df_results[df_results['Social Links Found'] > 0])
                success_rate = (sites_with_social / len(results) * 100) if results else 0
                total_links = df_results['Social Links Found'].sum()
                
                with col1:
                    st.metric("Sites Processed", len(results))
                with col2:
                    st.metric("Sites with Social Links", sites_with_social)
                with col3:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col4:
                    st.metric("Total Links Found", total_links)
                
                # Platform breakdown
                platform_counts = {}
                for _, row in df_results.iterrows():
                    platforms = row['Platforms'].split(', ') if row['Platforms'] else []
                    for platform in platforms:
                        if platform.strip():
                            platform_counts[platform.strip()] = platform_counts.get(platform.strip(), 0) + 1
                
                if platform_counts:
                    st.markdown("### 📊 Platform Distribution")
                    platform_df = pd.DataFrame(list(platform_counts.items()), columns=['Platform', 'Count'])
                    platform_df = platform_df.sort_values('Count', ascending=False)
                    st.bar_chart(platform_df.set_index('Platform'))
                
                # Results table with filtering
                st.markdown("### 📋 Detailed Results")
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox("Filter by status:", 
                                               ["All", "Success", "No social links found"])
                with col2:
                    platform_filter = st.selectbox("Filter by platform:", 
                                                  ["All"] + sorted(platform_counts.keys()) if platform_counts else ["All"])
                
                # Apply filters
                filtered_df = df_results.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['Status'] == status_filter]
                if platform_filter != "All":
                    filtered_df = filtered_df[filtered_df['Platforms'].str.contains(platform_filter, na=False)]
                
                st.dataframe(filtered_df, use_container_width=True)
                
                # Export options
                st.markdown("### 💾 Export Results")
                
                try:
                    export_files = export_results(results, export_format)
                    
                    for file_path in export_files:
                        with open(file_path, 'rb') as f:
                            file_ext = file_path.split('.')[-1]
                            mime_type = "text/csv" if file_ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            
                            st.download_button(
                                label=f"📄 Download {file_ext.upper()}",
                                data=f.read(),
                                file_name=file_path,
                                mime=mime_type
                            )
                
                except Exception as e:
                    st.error(f"❌ Error exporting results: {e}")
                
            except Exception as e:
                st.error(f"❌ Error during scraping: {e}")
                logging.error(f"Scraping error: {e}")

if __name__ == "__main__":
    main()