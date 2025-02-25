import json
import requests
import time
import os
from pathlib import Path

def download_abstract(arxiv_id):
    """Download abstract for a given arXiv ID."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "id_list": arxiv_id,
        "max_results": 1
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_arxiv_id(url):
    """Extract arXiv ID from URL."""
    if "arxiv.org/abs/" in url:
        return url.split("arxiv.org/abs/")[1]
    return None

def process_papers(input_file, output_dir):
    """Process papers and download abstracts."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load paper data
    with open(input_file, 'r') as f:
        papers = json.load(f)
    
    # Process top 10 papers
    for i, paper in enumerate(papers[:10]):
        title = paper['title']
        link = paper['link']
        
        print(f"Processing paper {i+1}: {title}")
        
        # Extract arXiv ID
        arxiv_id = extract_arxiv_id(link)
        if not arxiv_id:
            print(f"  Could not extract arXiv ID from {link}")
            continue
        
        # Download abstract
        response_text = download_abstract(arxiv_id)
        if not response_text:
            print(f"  Failed to download abstract for {arxiv_id}")
            continue
        
        # Save abstract
        output_file = Path(output_dir) / f"abstract_{arxiv_id.replace('/', '_')}.xml"
        with open(output_file, 'w') as f:
            f.write(response_text)
        
        print(f"  Saved abstract to {output_file}")
        
        # Be nice to the API
        time.sleep(3)

if __name__ == "__main__":
    input_file = "research/papers/paper_results.json"
    output_dir = "research/papers/abstracts"
    
    process_papers(input_file, output_dir)
