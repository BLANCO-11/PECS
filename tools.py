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