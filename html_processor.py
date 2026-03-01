"""HTML processing for URL-based paper reading.

Fetches URLs, detects content type (PDF vs HTML), parses HTML papers
into virtual pages with the same `--- Page N ---` format used by PDFs.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment

# Virtual page size (~3000 chars, split at paragraph/heading boundaries)
VIRTUAL_PAGE_SIZE = 3000

# Text truncation limits (same as pdf_processor)
MAX_TEXT_CHARS = 60000
KEEP_HEAD = 45000
KEEP_TAIL = 15000


@dataclass
class ProcessedHTML:
    title: str
    authors: str
    abstract: str
    full_text: str  # "--- Page N ---" formatted virtual pages
    num_pages: int
    clean_html: str  # sanitized HTML for iframe display
    figures: list  # list of dicts with fig_index, filename, page_num, etc.


# ---------------------------------------------------------------------------
# URL Fetching
# ---------------------------------------------------------------------------

def fetch_url(url: str) -> tuple[str, bytes]:
    """Fetch a URL and return (content_type, body_bytes).

    Uses a session with retries to handle sites that require cookies
    (e.g., anti-bot cookie checks on first visit).
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    netloc = parsed.netloc.lower()

    # Use domain-appropriate Referer
    if "weixin.qq.com" in netloc or "mp.weixin.qq.com" in netloc:
        referer = "https://mp.weixin.qq.com/"
    elif "zhihu.com" in netloc:
        referer = "https://www.zhihu.com/"
    else:
        referer = "https://www.google.com/"

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": referer,
    })

    # First request: may fail (403) but sets cookies needed for second try
    resp = session.get(url, timeout=60, allow_redirects=True)

    if resp.status_code == 403 or (
        resp.status_code == 200
        and "zhihu.com" in netloc
        and len(resp.content) < 5000
        and b"js_unavailable" in resp.content
    ):
        # Some sites set a challenge cookie on first visit; retry with cookies
        # Try hitting the origin first to get cookies, then the actual URL
        try:
            session.get(origin, timeout=15, allow_redirects=True)
        except Exception:
            pass
        resp = session.get(url, timeout=60, allow_redirects=True)

    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    return content_type, resp.content


def detect_content_type(url: str, content_type: str, body: bytes) -> str:
    """Determine if the URL points to a PDF or HTML document."""
    # Check Content-Type header
    ct_lower = content_type.lower()
    if "application/pdf" in ct_lower:
        return "pdf"
    if "text/html" in ct_lower or "application/xhtml" in ct_lower:
        return "html"

    # Check URL patterns
    url_lower = url.lower()
    if url_lower.endswith(".pdf") or "/pdf/" in url_lower:
        return "pdf"

    # Check magic bytes
    if body[:5] == b"%PDF-":
        return "pdf"

    # Default to HTML
    return "html"


# ---------------------------------------------------------------------------
# HTML Parsing
# ---------------------------------------------------------------------------

def _extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Find the main article content element."""
    # Try common academic paper containers (ordered most-specific first)
    for selector in [
        # WeChat public account articles
        "#js_content",
        ".rich_media_content",
        # Zhihu articles
        ".Post-RichTextContainer",
        ".RichText.ztext",
        ".Post-RichText",
        # Generic academic / blog
        "article",
        ".ltx_document",
        ".ltx_page_main",
        "main",
        '[role="main"]',
        "#content",
        ".content",
        ".paper-content",
    ]:
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) > 200:
            return el
    return soup.body or soup


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract paper title from HTML."""
    # WeChat public account title
    wechat_title = soup.select_one(".rich_media_title, #activity-name")
    if wechat_title:
        t = wechat_title.get_text(strip=True)
        if t:
            return t[:300]

    # Zhihu article title
    zhihu_title = soup.select_one(".Post-Title, .ContentItem-title")
    if zhihu_title:
        t = zhihu_title.get_text(strip=True)
        if t:
            return t[:300]

    # Try <h1> first
    h1 = soup.find("h1")
    if h1 and len(h1.get_text(strip=True)) > 3:
        return h1.get_text(strip=True)[:300]

    # Try meta tags
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in ("title", "dc.title", "citation_title", "og:title"):
            val = (meta.get("content") or "").strip()
            if val:
                return val[:300]

    # Try <title>
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)[:300]

    return "Untitled"


def _extract_authors(soup: BeautifulSoup) -> str:
    """Extract authors from meta tags or author elements."""
    authors = []

    # WeChat public account author/source
    wechat_author = soup.select_one(
        ".rich_media_meta_nickname, #js_name, .rich_media_meta_text"
    )
    if wechat_author:
        t = wechat_author.get_text(strip=True)
        if t:
            return t[:200]

    # Zhihu author
    zhihu_author = soup.select_one(".AuthorInfo-name, .UserLink-link")
    if zhihu_author:
        t = zhihu_author.get_text(strip=True)
        if t:
            return t[:200]

    # Meta tags
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in ("author", "dc.creator", "citation_author"):
            val = (meta.get("content") or "").strip()
            if val:
                authors.append(val)

    if authors:
        return ", ".join(authors)

    # Author elements (common in arXiv HTML)
    for cls in ["ltx_personname", "author", "authors"]:
        els = soup.find_all(class_=cls)
        for el in els:
            text = el.get_text(strip=True)
            if text and len(text) < 200:
                authors.append(text)
        if authors:
            break

    return ", ".join(authors[:20])


def _extract_abstract(soup: BeautifulSoup) -> str:
    """Extract abstract text."""
    # Look for abstract section
    for cls in ["ltx_abstract", "abstract"]:
        el = soup.find(class_=cls)
        if el:
            return el.get_text(strip=True)[:3000]

    # Look for heading containing "Abstract"
    for heading in soup.find_all(re.compile(r"^h[1-6]$", re.I)):
        if "abstract" in heading.get_text(strip=True).lower():
            # Get next sibling content
            parts = []
            for sib in heading.next_siblings:
                if hasattr(sib, "name") and sib.name and re.match(r"^h[1-6]$", sib.name):
                    break
                text = sib.get_text(strip=True) if hasattr(sib, "get_text") else str(sib).strip()
                if text:
                    parts.append(text)
            if parts:
                return " ".join(parts)[:3000]

    return ""


def _get_clean_text(el) -> str:
    """Get clean text from an element, collapsing whitespace."""
    text = el.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_virtual_pages(text: str) -> list[str]:
    """Split text into virtual pages of ~VIRTUAL_PAGE_SIZE chars.

    Splits at paragraph/heading boundaries (double newlines).
    """
    if not text:
        return [""]

    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n{2,}", text)
    pages = []
    current_page = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed limit and we have content,
        # start a new page
        if current_len + len(para) > VIRTUAL_PAGE_SIZE and current_page:
            pages.append("\n\n".join(current_page))
            current_page = []
            current_len = 0

        current_page.append(para)
        current_len += len(para)

    # Don't forget the last page
    if current_page:
        pages.append("\n\n".join(current_page))

    return pages if pages else [""]


def _truncate_text(text: str) -> str:
    """Truncate very long text (same logic as pdf_processor)."""
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return (
        text[:KEEP_HEAD]
        + "\n\n[... middle content truncated for length ...]\n\n"
        + text[-KEEP_TAIL:]
    )


# ---------------------------------------------------------------------------
# Figure Extraction
# ---------------------------------------------------------------------------

def _get_img_src(img) -> str:
    """Get the real image URL, handling lazy-load data-src attributes."""
    # Many sites (WeChat, Zhihu, etc.) lazy-load images via data-src
    for attr in ["data-src", "data-original", "data-lazy-src", "src"]:
        val = img.get(attr, "").strip()
        if val and not val.startswith("data:"):
            return val
    return ""


def _extract_figures(soup: BeautifulSoup, base_url: str, fig_dir: Path) -> list[dict]:
    """Extract figures from HTML: download images and collect captions."""
    figures = []
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_idx = 0

    # Look for <figure> elements
    for fig_el in soup.find_all("figure"):
        img = fig_el.find("img")
        if not img:
            continue
        src = _get_img_src(img)
        if not src:
            continue

        caption_el = fig_el.find("figcaption")
        caption = caption_el.get_text(strip=True)[:200] if caption_el else ""

        img_url = urljoin(base_url, src)
        fig_idx += 1
        filename = f"fig_{fig_idx}.png"

        try:
            resp = requests.get(img_url, timeout=30)
            resp.raise_for_status()
            (fig_dir / filename).write_bytes(resp.content)
            figures.append({
                "fig_index": fig_idx,
                "filename": filename,
                "page_num": 0,
                "width": 400,
                "height": 300,
                "caption": caption or f"Figure {fig_idx}",
            })
        except Exception:
            continue

    # Also look for standalone <img> with meaningful alt/data-src if no <figure> found
    if not figures:
        for img in soup.find_all("img"):
            src = _get_img_src(img)
            alt = img.get("alt", "")
            if not src or len(alt) < 3:
                continue
            # Skip tiny icons/decorations
            width = img.get("width", "")
            if width and str(width).isdigit() and int(width) < 50:
                continue

            img_url = urljoin(base_url, src)
            fig_idx += 1
            filename = f"fig_{fig_idx}.png"

            try:
                resp = requests.get(img_url, timeout=30)
                resp.raise_for_status()
                (fig_dir / filename).write_bytes(resp.content)
                figures.append({
                    "fig_index": fig_idx,
                    "filename": filename,
                    "page_num": 0,
                    "width": 400,
                    "height": 300,
                    "caption": alt[:200],
                })
            except Exception:
                continue

    return figures


# ---------------------------------------------------------------------------
# Clean HTML Generation
# ---------------------------------------------------------------------------

_HIGHLIGHT_SCRIPT = """
<script>
(function() {
  window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'highlight') return;
    var page = e.data.page;
    var quote = e.data.quote;
    var yNorm = e.data.yNorm;

    // Remove previous highlights
    document.querySelectorAll('.dr-highlight').forEach(function(el) {
      var parent = el.parentNode;
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    });

    // Try to find and highlight the quote text
    if (quote) {
      var found = highlightText(quote);
      if (found) return;
    }

    // Fallback: scroll to virtual page section
    var pageEl = document.querySelector('[data-page="' + page + '"]');
    if (pageEl) {
      var yOffset = 0;
      if (yNorm !== null && yNorm !== undefined) {
        yOffset = (yNorm / 1000) * pageEl.offsetHeight;
      }
      var targetY = pageEl.offsetTop + yOffset - window.innerHeight / 3;
      window.scrollTo({ top: targetY, behavior: 'smooth' });
    }
  });

  function highlightText(quote) {
    var walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );
    var fullText = '';
    var textNodes = [];
    var node;
    while (node = walker.nextNode()) {
      textNodes.push({ node: node, start: fullText.length });
      fullText += node.textContent;
    }

    var searchLower = quote.toLowerCase();
    var fullLower = fullText.toLowerCase();
    var idx = fullLower.indexOf(searchLower);

    // Flexible whitespace match
    if (idx === -1) {
      var words = searchLower.split(/\\s+/).filter(function(w) { return w.length > 0; });
      if (words.length >= 2) {
        var pattern = words.map(function(w) {
          return w.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        }).join('\\\\s+');
        var regex = new RegExp(pattern, 'i');
        var m = fullText.match(regex);
        if (m) { idx = m.index; searchLower = m[0].toLowerCase(); }
      }
    }

    if (idx === -1) return false;

    var matchEnd = idx + searchLower.length;

    // Find and wrap matching text nodes
    var firstHighlight = null;
    for (var i = 0; i < textNodes.length; i++) {
      var tn = textNodes[i];
      var nodeEnd = tn.start + tn.node.textContent.length;
      if (tn.start >= matchEnd || nodeEnd <= idx) continue;

      var startInNode = Math.max(0, idx - tn.start);
      var endInNode = Math.min(tn.node.textContent.length, matchEnd - tn.start);

      var before = tn.node.textContent.substring(0, startInNode);
      var match = tn.node.textContent.substring(startInNode, endInNode);
      var after = tn.node.textContent.substring(endInNode);

      var mark = document.createElement('mark');
      mark.className = 'dr-highlight';
      mark.style.background = 'rgba(255, 212, 0, 0.5)';
      mark.style.padding = '1px 0';
      mark.style.borderRadius = '2px';
      mark.textContent = match;

      var parent = tn.node.parentNode;
      if (after) parent.insertBefore(document.createTextNode(after), tn.node.nextSibling);
      parent.insertBefore(mark, tn.node.nextSibling);
      if (before) {
        tn.node.textContent = before;
      } else {
        parent.removeChild(tn.node);
      }

      if (!firstHighlight) firstHighlight = mark;
    }

    if (firstHighlight) {
      var rect = firstHighlight.getBoundingClientRect();
      window.scrollTo({
        top: window.scrollY + rect.top - window.innerHeight / 3,
        behavior: 'smooth'
      });

      // Auto-fade after 5 seconds
      setTimeout(function() {
        document.querySelectorAll('.dr-highlight').forEach(function(el) {
          el.style.transition = 'background 1.5s ease-out';
          el.style.background = 'transparent';
          setTimeout(function() {
            var p = el.parentNode;
            p.replaceChild(document.createTextNode(el.textContent), el);
            p.normalize();
          }, 2000);
        });
      }, 5000);

      return true;
    }
    return false;
  }
})();
</script>
"""


def _has_latex_math(html_str: str) -> bool:
    """Detect if the HTML contains LaTeX math content that needs rendering."""
    indicators = [
        r"\begin{equation",
        r"\begin{align",
        r"\begin{gather",
        r"\begin{eqnarray",
        r"\begin{multline",
        r"\[",
        r"\(",
    ]
    for ind in indicators:
        if ind in html_str:
            return True
    # Check for $$...$$ or $...$ patterns (but avoid false positives from currency)
    if re.search(r'\$\$[^$]+\$\$', html_str):
        return True
    if re.search(r'(?<!\w)\$[^$\n]{3,}\$(?!\w)', html_str):
        return True
    return False


# MathJax 3 CDN injection — renders LaTeX math in the displayed HTML
_MATHJAX_INJECT = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    tags: 'ams'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  startup: {
    pageReady: function() {
      return MathJax.startup.defaultPageReady().then(function() {
        // Signal parent that math is rendered (for potential future use)
        window.parent.postMessage({type: 'mathRendered'}, '*');
      });
    }
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
"""


def _build_clean_html(
    soup: BeautifulSoup,
    base_url: str,
    virtual_pages: list[str],
    full_text: str,
) -> str:
    """Build display HTML that preserves original page appearance.

    Strategy: keep the full original HTML (styles, layout, fonts) intact.
    Only remove <script> tags for security, convert relative URLs to absolute,
    and inject our highlight/postMessage handler script.
    If the page contains LaTeX math, inject MathJax 3 from CDN.
    """
    doc = soup

    # Snapshot: detect math before removing scripts (scripts may contain
    # MathJax config which is itself a signal, but raw LaTeX in body is
    # the definitive indicator)
    needs_math = _has_latex_math(str(doc))

    # 1. Remove all <script> tags (security — iframe sandboxed)
    for tag in doc.find_all("script"):
        tag.decompose()
    # Remove noscript wrappers (show their content)
    for tag in doc.find_all("noscript"):
        tag.unwrap()
    # Remove HTML comments
    for comment in doc.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 2. Un-hide JS-revealed content: remove inline visibility:hidden / display:none
    # WeChat sets #js_content { visibility: hidden } and JS removes it.
    # Since we strip scripts, we force-reveal any element hidden this way
    # that actually contains text content.
    for tag in doc.find_all(attrs={"style": True}):
        style = tag.get("style", "")
        new_style = style
        if re.search(r'visibility\s*:\s*hidden', style, re.I):
            new_style = re.sub(r'visibility\s*:\s*hidden', 'visibility:visible', new_style, flags=re.I)
        if re.search(r'display\s*:\s*none', new_style, re.I):
            # Only un-hide if the element has substantial text (avoids menus etc.)
            if len(tag.get_text(strip=True)) > 100:
                new_style = re.sub(r'display\s*:\s*none', 'display:block', new_style, flags=re.I)
        if new_style != style:
            tag["style"] = new_style

    # 3. Convert relative URLs to absolute for all resource types
    for tag in doc.find_all(["img", "a", "link", "source", "video", "audio"]):
        # Promote data-src / data-original (lazy-load) to src for images
        if tag.name == "img":
            for lazy_attr in ["data-src", "data-original", "data-lazy-src"]:
                lazy_val = tag.get(lazy_attr, "").strip()
                if lazy_val and not lazy_val.startswith("data:"):
                    tag["src"] = lazy_val
                    break
        for attr in ["src", "href", "poster"]:
            val = tag.get(attr)
            if val and not val.startswith(("http://", "https://", "data:", "#", "javascript:")):
                tag[attr] = urljoin(base_url, val)
    # Also fix srcset attributes
    for tag in doc.find_all(attrs={"srcset": True}):
        parts = []
        for entry in tag["srcset"].split(","):
            entry = entry.strip()
            if not entry:
                continue
            tokens = entry.split()
            if tokens and not tokens[0].startswith(("http://", "https://", "data:")):
                tokens[0] = urljoin(base_url, tokens[0])
            parts.append(" ".join(tokens))
        tag["srcset"] = ", ".join(parts)
    # Fix CSS url() in inline styles
    for tag in doc.find_all(attrs={"style": True}):
        style = tag["style"]
        if "url(" in style:
            def _fix_css_url(m):
                url_val = m.group(1).strip("'\"")
                if not url_val.startswith(("http://", "https://", "data:")):
                    url_val = urljoin(base_url, url_val)
                return f"url('{url_val}')"
            tag["style"] = re.sub(r"url\(([^)]+)\)", _fix_css_url, style)

    # 4. Inject <base> tag so any remaining relative URLs resolve correctly
    head = doc.find("head")
    if head:
        # Remove existing <base> if any
        for existing_base in head.find_all("base"):
            existing_base.decompose()
        base_tag = doc.new_tag("base", href=base_url)
        head.insert(0, base_tag)
    else:
        # No <head>, create one
        html_tag = doc.find("html")
        if html_tag:
            head = doc.new_tag("head")
            head.append(doc.new_tag("base", href=base_url))
            html_tag.insert(0, head)

    # 5. Inject scripts before </body>
    body = doc.find("body")
    if body:
        # Inject MathJax if the page has LaTeX math
        if needs_math:
            math_soup = BeautifulSoup(_MATHJAX_INJECT, "html.parser")
            body.append(math_soup)

        # Inject highlight/postMessage handler
        script_soup = BeautifulSoup(_HIGHLIGHT_SCRIPT, "html.parser")
        body.append(script_soup)

    return str(doc)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def process_html(url: str, html_bytes: bytes, paper_id: str, fig_dir: str) -> ProcessedHTML:
    """Process an HTML page into virtual pages for DeepReading.

    Returns a ProcessedHTML with the same text format as PDF processing.
    """
    # Detect encoding
    encoding = "utf-8"
    try:
        html_str = html_bytes.decode(encoding)
    except UnicodeDecodeError:
        html_str = html_bytes.decode("latin-1")

    soup = BeautifulSoup(html_str, "lxml")

    # Extract metadata
    title = _extract_title(soup)
    authors = _extract_authors(soup)
    abstract_text = _extract_abstract(soup)

    # Extract main text content
    main_content = _extract_main_content(soup)
    # Get paragraphs and headings as separate blocks
    text_blocks = []
    for el in main_content.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "figcaption", "dt", "dd"]
    ):
        text = _get_clean_text(el)
        if text and len(text) > 1:
            # Add heading markers
            if el.name and el.name.startswith("h"):
                text = "\n" + text
            text_blocks.append(text)

    # If no structured blocks found, fall back to full text
    if not text_blocks:
        text_blocks = [_get_clean_text(main_content)]

    full_plain_text = "\n\n".join(text_blocks)

    # Split into virtual pages
    virtual_pages = _split_into_virtual_pages(full_plain_text)

    # Format as --- Page N --- (same as PDF processor)
    pages_text = []
    for i, page_content in enumerate(virtual_pages):
        pages_text.append(f"--- Page {i + 1} ---\n{page_content}")
    full_text = "\n".join(pages_text)
    full_text = _truncate_text(full_text)

    # Extract figures
    fig_path = Path(fig_dir)
    figures = _extract_figures(soup, url, fig_path)

    # Build clean HTML for iframe display — re-parse from original bytes
    # so the soup used for text extraction above is not affected
    display_soup = BeautifulSoup(html_str, "lxml")
    clean_html = _build_clean_html(display_soup, url, virtual_pages, full_plain_text)

    return ProcessedHTML(
        title=title,
        authors=authors,
        abstract=abstract_text,
        full_text=full_text,
        num_pages=len(virtual_pages),
        clean_html=clean_html,
        figures=figures,
    )
