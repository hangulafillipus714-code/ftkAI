import re
import os
import unicodedata

# ==========================================================
# Core Cleaning Functions
# ==========================================================

def remove_html_tags(text: str) -> str:
    """
    Removes HTML/XML tags from text.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_js_css(text: str) -> str:
    """
    Removes JavaScript and CSS blocks from text.
    """
    # Remove script tags
    text = re.sub(r"<script\b[^>]*>.*?<\/script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove style tags
    text = re.sub(r"<style\b[^>]*>.*?<\/style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove event handlers (e.g., onclick="...")
    text = re.sub(r"on[a-z]+\s*=\s*\"[^\"]*\"", "", text, flags=re.IGNORECASE)
    text = re.sub(r"on[a-z]+\s*=\s*'[^']*'", "", text, flags=re.IGNORECASE)
    # Remove CSS-like attributes (e.g., style="...")
    text = re.sub(r"style\s*=\s*\"[^\"]*\"", "", text, flags=re.IGNORECASE)
    text = re.sub(r"style\s*=\s*'[^']*'", "", text, flags=re.IGNORECASE)
    return text

def generic_clean(content: str) -> str:
    """
    Applies a series of robust generic cleaning steps to text data.
    """
    # 1. Normalize Unicode characters to their closest ASCII equivalent
    content = unicodedata.normalize('NFKC', content)

    # 2. Remove non-printable ASCII characters (excluding basic whitespace)
    content = ''.join(char for char in content if 31 < ord(char) < 127 or char in '\n\t ')

    # 3. Replace common typographic quotes with straight quotes
    content = content.replace('“', '"').replace('”', '"')
    content = content.replace('‘', ''').replace('’', ''')
    content = content.replace('—', '--').replace('–', '-')
    content = content.replace('…', '...')

    # 4. Remove URLs
    content = re.sub(r'https?:\/\/\S+', '', content)
    content = re.sub(r'www\.\S+', '', content)

    # 5. Normalize whitespace: multiple spaces to single space
    content = re.sub(r'[ 	]+', ' ', content)

    # 6. Reduce excessive newlines to at most two consecutive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 7. Remove leading/trailing whitespace from each line and then from the whole content
    content = '\n'.join([line.strip() for line in content.split('\n')])
    content = content.strip()

    # 8. Remove lines that are purely whitespace or very short and likely meaningless
    lines = content.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip().split()) > 1 or len(line.strip()) > 5]
    content = '\n'.join(cleaned_lines)
    
    # 9. Deduplicate consecutive identical lines (case-insensitive for efficiency)
    deduped_lines: list[str] = []
    for line in content.split('\n'):
        if not deduped_lines or line.lower().strip() != deduped_lines[-1].lower().strip():
            deduped_lines.append(line)
    content = '\n'.join(deduped_lines)

    return content.strip()

def clean_gutenberg(content: str) -> str:
    """
    Cleans Project Gutenberg specific headers, footers, and metadata.
    """
    # Apply generic cleaning first
    content = generic_clean(content)

    # Standard Gutenberg markers
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", content)
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", content)
    
    if start_match and end_match:
        content = content[start_match.end():end_match.start()]
    elif start_match:
        content = content[start_match.end():]
    elif end_match:
        content = content[:end_match.start()]
    
    # Remove "Other editions" blurbs and file number lists
    content = re.sub(r"There are several editions of this ebook.*?preferred file\.", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"Click on any of the filenumbers below.*?(\d{4,}\s+.*?(\n|$))+", "", content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove Table of Contents and Indices
    toc_pattern = r"\b(Contents|INDEX|DETAILED|TABLE OF CONTENTS)\b.*?(?:\s*\d+\.\s+.*?\n|\s*[A-Z ]+\.\s+.*?\n|\s*Chapter\s+\d+.*?\n|\s*Letter\s+\d+.*?\n|\s*.*?\.\.\..*?\n)+"
    content = re.sub(toc_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

    # Remove specific title page blocks (e.g., Darwin's title/address)
    content = re.sub(r"On\s+the\s+Origin\s+of\s+Species.*?October_,\s+1_st_,\s+1859\.", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"BY\s+MEANS\s+OF\s+NATURAL\s+SELECTION.*?ALBEMARLE\s+STREET\.", "", content, flags=re.DOTALL | re.IGNORECASE)

    # Remove excessive title page metadata (Title, Author, Release Date, etc.)
    metadata_patterns = [
        r"^Title:.*?\n",
        r"^Author:.*?\n",
        r"^Release date:.*?\n",
        r"^Language:.*?\n",
        r"^Most recently updated:.*?\n",
        r"^Character set encoding:.*?\n",
        r"^Produced by:.*?\n",
        r"^\[.*?\]\n"  # Bracketed notes
    ]
    for pattern in metadata_patterns:
        content = re.sub(pattern, "", content, flags=re.MULTILINE | re.IGNORECASE)

    return content.strip()

def clean_web_text(content: str) -> str:
    """
    Cleans web-scraped text by removing common noise elements and short, irrelevant lines.
    """
    # Apply generic cleaning first
    content = generic_clean(content)

    lines = content.split('\n')
    cleaned_lines = []
    
    # Common web noise patterns
    noise_keywords = {
        'skip to content', 'skip to main content', 'working...', 'menu', 'facebook', 'twitter', 
        'privacy policy', 'terms of service', 'copyright', 'all rights reserved', 'contact us',
        'about us', 'search', 'login', 'sign up', 'navigation', 'back to top', 'similar domains',
        'powered by', 'daily visitors', 'pageviews', 'rank', 'ping statistics', 'benefits & services',
        'alumni a-card', 'financial services', 'merchandise', 'learn and explore', 'events',
        'social media', 'reunions', 'where you live', 'faculties & schools', 'address update',
        'subscribe', 'newsletter', 'ad privacy', 'cookie settings', 'manage preferences',
        'privacy preference center', 'accept all cookies', 'opt out', 'get started',
        'read more', 'view all', 'download now', 'share this article', 'related articles',
        'latest news', 'recent posts', 'blog post', 'comments', 'leave a comment',
        'your email address will not be published', 'required fields are marked',
        'post comment', 'cancel reply', 'website', 'name', 'email', 'save my name',
        'next page', 'previous page', 'next post', 'previous post',
        'home', 'blog', 'categories', 'tags', 'archive', 'sitemap',
    }
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Filter out lines containing noise keywords
        if any(keyword in stripped.lower() for keyword in noise_keywords):
            continue
            
        # Filter out short navigation-like lines (less than 4 words)
        if len(stripped.split()) < 4:
            continue
        
        # Filter out technical log-like lines
        if (re.search(r'^\d+\s*(bytes from|ms|kb/s|B/s)', stripped) or 
            re.search(r'icmp_seq=', stripped) or 
            re.search(r'^IP: \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', stripped) or 
            re.search(r'^rtt min/avg/max/mdev', stripped) or 
            re.search(r'^Received \d+ bytes from', stripped, re.IGNORECASE)):
            continue
            
        cleaned_lines.append(stripped)
    
    return '\n'.join(cleaned_lines)
