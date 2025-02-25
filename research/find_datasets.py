import json
import re
from collections import Counter

def find_datasets(input_file):
    """Find mentions of common emotion datasets in papers."""
    # Common emotion datasets
    dataset_patterns = [
        r'\bIEMOCAP\b',
        r'\bMSP-IMPROV\b',
        r'\bRAVDESS\b',
        r'\bCREMA-D\b',
        r'\bEMOVO\b',
        r'\bEMODB\b',
        r'\bEMOVOME\b',
        r'\bSEMO\b',
        r'\bDEMOS\b',
        r'\bMEAD\b',
        r'\bAffectNet\b',
        r'\bFER\+\b',
        r'\bRAFDB\b',
        r'\bAffWild\b',
        r'\bMOSEI\b',
        r'\bMOSI\b',
        r'\bCMU-MOSEI\b',
        r'\bCMU-MOSI\b',
        r'\bDEAP\b',
        r'\bMAHNOB-HCI\b'
    ]
    
    # Load paper data
    with open(input_file, 'r') as f:
        papers = json.load(f)
    
    # Count dataset mentions
    dataset_counts = Counter()
    dataset_papers = {}
    
    for paper in papers:
        title = paper['title']
        abstract = paper['abstract']
        text = title + " " + abstract
        
        for pattern in dataset_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                dataset = matches[0].upper()
                dataset_counts[dataset] += len(matches)
                
                if dataset not in dataset_papers:
                    dataset_papers[dataset] = []
                
                dataset_papers[dataset].append({
                    'title': title,
                    'link': paper['link']
                })
    
    # Print results
    print("Dataset mentions in papers:")
    for dataset, count in dataset_counts.most_common():
        print(f"{dataset}: {count} mentions")
        print("  Papers:")
        for paper in dataset_papers[dataset][:3]:  # Show top 3 papers for each dataset
            print(f"  - {paper['title']}")
            print(f"    {paper['link']}")
        print()
    
    return dataset_counts, dataset_papers

if __name__ == "__main__":
    input_file = "research/papers/paper_results.json"
    find_datasets(input_file)
