import requests
import time
import json
import os
from pathlib import Path

def search_arxiv(query, max_results=10):
    """Search arXiv for papers matching the query."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        return response.text
    else:
        return None

def parse_arxiv_response(response_text):
    """Parse arXiv API response to extract paper details."""
    import xml.etree.ElementTree as ET
    
    root = ET.fromstring(response_text)
    
    # Define namespace
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    
    for entry in root.findall('.//atom:entry', ns):
        title = entry.find('./atom:title', ns).text.strip()
        
        # Get authors
        authors = []
        for author in entry.findall('./atom:author/atom:name', ns):
            authors.append(author.text.strip())
        
        # Get abstract
        abstract = entry.find('./atom:summary', ns).text.strip()
        
        # Get link
        link = entry.find('./atom:id', ns).text.strip()
        
        # Get published date
        published = entry.find('./atom:published', ns).text.strip()
        
        papers.append({
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'link': link,
            'published': published
        })
    
    return papers

def search_papers(queries, output_dir, max_results=10):
    """Search for papers using multiple queries and save results."""
    all_papers = []
    
    for query in queries:
        print(f"Searching for: {query}")
        response = search_arxiv(query, max_results)
        
        if response:
            papers = parse_arxiv_response(response)
            all_papers.extend(papers)
            print(f"Found {len(papers)} papers")
        else:
            print(f"Failed to get results for query: {query}")
        
        # Be nice to the API
        time.sleep(3)
    
    # Remove duplicates based on title
    unique_papers = []
    seen_titles = set()
    
    for paper in all_papers:
        if paper['title'] not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(paper['title'])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = Path(output_dir) / "paper_results.json"
    with open(output_file, 'w') as f:
        json.dump(unique_papers, f, indent=2)
    
    print(f"Saved {len(unique_papers)} unique papers to {output_file}")
    return unique_papers

if __name__ == "__main__":
    queries = [
        "all:emotion recognition audio valence arousal dominance",
        "all:speech emotion recognition valence arousal dominance",
        "all:audio emotion recognition dimensional model",
        "all:mapping valence arousal dominance to emotion categories",
        "all:speech emotion recognition regression valence arousal"
    ]
    
    output_dir = "research/papers"
    papers = search_papers(queries, output_dir, max_results=20)
    
    # Print top 5 papers
    print("\nTop 5 papers:")
    for i, paper in enumerate(papers[:5]):
        print(f"{i+1}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Link: {paper['link']}")
        print()
