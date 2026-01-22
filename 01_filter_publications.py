"""
Publication Filter for Researcher Profiling Pipeline

This script filters publications from a JSON file to keep only those where
the target researcher is a primary contributor (first 3 or last 3 author positions).

Usage:
    python 01_filter_publications.py <json_file> <researcher_name>

Example:
    python 01_filter_publications.py "data/input_sample/Chunhua Weng.json" "Chunhua Weng"
"""

import json
import csv
import sys
import os

# Configuration constants
PRIMARY_AUTHOR_THRESHOLD = 3  # Keep papers where researcher is in first/last N positions
DEFAULT_INPUT_FILE = "data/input_sample/Chunhua Weng.json"
DEFAULT_RESEARCHER_NAME = "Chunhua Weng"
DEFAULT_OUTPUT_PATH = "data/output_sample/preprocess/filtered_publications.csv"


def filter_publications(json_file_path, target_researcher_name, output_csv_path=DEFAULT_OUTPUT_PATH):
    """
    Filter publications to keep only those where the researcher is a primary contributor.

    Parameters
    ----------
    json_file_path : str
        Path to the JSON file containing publication records.
    target_researcher_name : str
        Full name of the researcher to filter for.
    output_csv_path : str
        Path where the filtered CSV will be saved.

    Returns
    -------
    None
        Outputs are written to the specified CSV file.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' not found.")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
        return

    valid_papers = []
    print(f"Scanning {len(data)} records for researcher: {target_researcher_name}...")

    # Step 1: Filter papers by author position
    for paper in data:
        authors_list = paper.get("Authors", [])
        researcher_index = _find_researcher_index(authors_list, target_researcher_name)

        if _is_primary_author(researcher_index, len(authors_list)):
            if paper.get("MeSH terms"):
                valid_papers.append(paper)

    # Step 2: Prepare researcher metadata
    unique_pmids = set(paper.get("PMID") for paper in valid_papers if paper.get("PMID"))
    num_publications = len(unique_pmids)

    target_first, target_last = _split_name(target_researcher_name)

    # Step 3: Write results to CSV
    if valid_papers:
        _write_output_csv(valid_papers, output_csv_path, target_first, target_last, num_publications)
        print(f"Success! Processed {num_publications} papers with MeSH terms.")
        print(f"Saved to: {output_csv_path}")
    else:
        print("No matching records found (checked author position AND presence of MeSH terms).")


def _find_researcher_index(authors_list, target_name):
    """Find the index of the target researcher in the author list."""
    for idx, author in enumerate(authors_list):
        first = author.get("First Name", "").strip()
        last = author.get("Last Name", "").strip()
        full_name = f"{first} {last}"

        if full_name.lower() == target_name.lower():
            return idx
    return -1


def _is_primary_author(researcher_index, total_authors):
    """
    Check if the researcher is in a primary author position.

    Primary positions are:
    - First N authors (typically first author, co-first authors)
    - Last N authors (typically senior/corresponding authors)
    """
    if researcher_index == -1:
        return False
    return (researcher_index < PRIMARY_AUTHOR_THRESHOLD or
            researcher_index >= total_authors - PRIMARY_AUTHOR_THRESHOLD)


def _split_name(full_name):
    """Split a full name into first and last name components."""
    name_parts = full_name.strip().split()
    if len(name_parts) >= 2:
        return name_parts[0], " ".join(name_parts[1:])
    return full_name, ""


def _write_output_csv(valid_papers, output_csv_path, first_name, last_name, num_publications):
    """Write filtered papers to CSV file."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    fieldnames = ["Person ID", "First Name", "Last Name", "Num Publications", "MeSH Term", "PMID"]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for paper in valid_papers:
            pmid = paper.get("PMID", "")
            mesh_terms = paper.get("MeSH terms", [])

            for term in mesh_terms:
                writer.writerow({
                    "Person ID": 1,
                    "First Name": first_name,
                    "Last Name": last_name,
                    "Num Publications": num_publications,
                    "MeSH Term": term,
                    "PMID": pmid
                })


if __name__ == "__main__":
    json_input = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    name_input = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RESEARCHER_NAME

    filter_publications(json_input, name_input)
