import json
import csv
import sys
import os


def filter_publications(json_file_path, target_researcher_name, output_csv_path="data/output_sample/preprocess/filtered_publications.csv"):
    # Check if file exists
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

    # Step 1: Filter and Collect Valid Papers (Author Position Logic)
    for paper in data:
        authors_list = paper.get("Authors", [])

        # Find researcher index
        researcher_index = -1
        for idx, author in enumerate(authors_list):
            first = author.get("First Name", "").strip()
            last = author.get("Last Name", "").strip()
            full_name = f"{first} {last}"

            if full_name.lower() == target_researcher_name.lower():
                researcher_index = idx
                break

        # Check Position: First 3 OR Last 3
        total_authors = len(authors_list)
        is_primary = False
        if researcher_index != -1:
            if researcher_index < 3 or researcher_index >= total_authors - 3:
                is_primary = True

        if is_primary:
            if paper.get("MeSH terms"):
                valid_papers.append(paper)

    # Step 2: Prepare Researcher Metadata
    # Calculate number of unique PMIDs
    unique_pmids = set(paper.get("PMID") for paper in valid_papers if paper.get("PMID"))
    num_publications = len(unique_pmids)

    # Split target name for columns
    name_parts = target_researcher_name.strip().split()
    if len(name_parts) >= 2:
        target_first = name_parts[0]
        target_last = " ".join(name_parts[1:])
    else:
        target_first = target_researcher_name
        target_last = ""

    # Step 3: Write to CSV
    if valid_papers:
        # EXACT Sequence requested
        fieldnames = ["Person ID", "First Name", "Last Name", "Num Publications", "MeSH Term", "PMID"]

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for paper in valid_papers:
                pmid = paper.get("PMID", "")
                mesh_terms = paper.get("MeSH terms", [])

                # Since we pre-filtered valid_papers to only include those with MeSH terms,
                # we can just loop directly.
                for term in mesh_terms:
                    writer.writerow({
                        "Person ID": 1,
                        "First Name": target_first,
                        "Last Name": target_last,
                        "Num Publications": num_publications,
                        "MeSH Term": term,
                        "PMID": pmid
                    })

        print(f"Success! Processed {num_publications} papers with MeSH terms.")
        print(f"Saved to: {output_csv_path}")
    else:
        print("No matching records found (checked author position AND presence of MeSH terms).")


if __name__ == "__main__":
    # Default values
    default_file = "data/input_sample/Chunhua Weng.json"
    default_name = "Chunhua Weng"

    # Allow command line arguments
    json_input = sys.argv[1] if len(sys.argv) > 1 else default_file
    name_input = sys.argv[2] if len(sys.argv) > 2 else default_name

    filter_publications(json_input, name_input)