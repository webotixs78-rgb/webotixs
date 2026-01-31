# socialscraper.py
# Streamlit-safe synchronous version (NO asyncio, NO aiohttp)

import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

# -------------------------
# Config
# -------------------------
SOCIAL_DOMAINS = ["facebook.com", "instagram.com"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# -------------------------
# HTTP Fetch (SYNC)
# -------------------------
def fetch(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.text
        return ""
    except Exception:
        return ""

# -------------------------
# Extract links
# -------------------------
def extract_links_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        links.add(href)

    return links

# -------------------------
# Filter social links
# -------------------------
def filter_social_links(links):
    return list(
        set(
            link for link in links
            if any(domain in link.lower() for domain in SOCIAL_DOMAINS)
        )
    )

# -------------------------
# Crawl a single website
# -------------------------
def crawl_site(base_url):
    visited = set()
    to_visit = {base_url}
    social_links_found = {}

    for _ in range(3):  # crawl max 3 pages
        if not to_visit:
            break

        current_url = to_visit.pop()
        visited.add(current_url)

        html = fetch(current_url)
        if not html:
            continue

        links = extract_links_from_html(html, base_url)
        social_links = filter_social_links(links)

        for social in social_links:
            if social not in social_links_found:
                social_links_found[social] = current_url

        if social_links_found:
            break

        for link in links:
            if base_url in link and link not in visited:
                to_visit.add(link)

    result = {"Website": base_url}

    if social_links_found:
        for i, (social, source) in enumerate(social_links_found.items(), start=1):
            result[f"Social Link {i}"] = social
            result[f"Found On Page {i}"] = source
    else:
        result["Social Link 1"] = ""
        result["Found On Page 1"] = ""

    return result

# -------------------------
# Process websites
# -------------------------
def process_websites(websites):
    results = []
    progress_bar = st.progress(0)

    for i, site in enumerate(websites):
        results.append(crawl_site(site))
        progress_bar.progress((i + 1) / len(websites))

    return results

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Social Media Link Scraper",
    layout="centered"
)

st.title("🔍 Website Social Media Link Scraper")

st.markdown("""
### 📋 Instructions
- Upload a CSV file with a **`Website`** column  
- Example: `https://example.com`  
- The app will find **Facebook & Instagram** links  
- Crawls up to **3 pages per website**
""")

uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Website" not in df.columns:
        st.error("❌ CSV must contain a 'Website' column")
    else:
        websites = df["Website"].dropna().tolist()

        if st.button("🚀 Start Scraping"):
            with st.spinner("Scraping websites..."):
                results = process_websites(websites)

            result_df = pd.DataFrame(results)
            st.success("✅ Scraping completed!")

            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name="social_links_results.csv",
                mime="text/csv"
            )
