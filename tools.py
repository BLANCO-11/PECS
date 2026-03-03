import urllib.request
import urllib.parse
import json
import re

def fetch_rss_news(feed_url="http://feeds.bbci.co.uk/news/world/rss.xml"):
    """Fetches top news from a public RSS feed."""
    try:
        import xml.etree.ElementTree as ET # Import here to keep module clean
        req = urllib.request.Request(
            feed_url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        # Set a timeout to prevent hanging
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = []
            # Fetch all items
            for item in root.findall('.//item'): 
                title = item.find('title').text
                link = item.find('link').text
                desc = item.find('description').text
                items.append({'title': title, 'link': link, 'summary': desc})
            return items
    except Exception as e:
        print(f"Error fetching RSS: {e}")
        return []

def search_wikipedia(topic: str):
    """
    Searches Wikipedia for a topic and returns the summary of the top result.
    Returns: (url, summary_text) or (None, None)
    """
    try:
        headers = {'User-Agent': 'AlphaCognitiveCore/1.0 (https://github.com/BLANCO-11/alpha-core; saini77aku@gmail.com)'}
        
        # 1. Search for the page title using list=search (more robust than opensearch)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(topic)}&srlimit=1&format=json"
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            if not data.get('query', {}).get('search'):
                return None, None
            
            title = data['query']['search'][0]['title']
            # Construct URL for display
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            
        # 2. Fetch the summary (extract) for that title
        # using query action to get plain text extract
        query_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(title)}&format=json"
        req = urllib.request.Request(query_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            
            if page_id == "-1":
                return None, None
                
            summary = pages[page_id].get('extract', "")
            return url, summary

    except Exception as e:
        print(f"Error searching Wikipedia: {e}")
        return None, None

def search_web(query: str, max_results: int = 5):
    """
    Searches the web using DuckDuckGo and returns the top results.
    """
    results = []
    try:
        from ddgs import DDGS
        with DDGS(timeout=10) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r.get('title'),
                    'link': r.get('href'),
                    'summary': r.get('body')
                })
        return results
    except Exception as e:
        print(f"[Tools] DuckDuckGo error: {e}")
        return []

def fetch_webpage_text(url: str) -> str:
    """
    Fetches the main text content from a webpage URL using Readability (if available) or BeautifulSoup.
    """
    try:
        from bs4 import BeautifulSoup, GuessedAtParserWarning
        import warnings
        # Suppress the parser warning since we are okay with the default
        warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

        req = urllib.request.Request(
            url,
            data=None,
            # A realistic user agent is important for many sites
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            if 'text/html' not in response.getheader('Content-Type', ''):
                return ""
            
            html_content = response.read()
            
            # Try readability first
            try:
                from readability import Document
                doc = Document(html_content)
                summary_html = doc.summary()
                soup = BeautifulSoup(summary_html, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                return text
            except ImportError:
                pass
            except Exception:
                pass # Fallback to raw BS4

            soup = BeautifulSoup(html_content, 'html.parser')

            for tag in soup(['nav', 'header', 'footer', 'aside', 'script', 'style', 'form']):
                tag.decompose()

            text = soup.get_text(separator='\n', strip=True)
            lines = (line.strip() for line in text.splitlines())
            non_empty_lines = [line for line in lines if line]
            
            return "\n".join(non_empty_lines)

    except Exception as e:
        print(f"Error fetching webpage content for {url}: {e}")
        return ""