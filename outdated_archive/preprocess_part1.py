import json 
import os
import pandas as pd
import re

file_dir  = "test_names/Chunhua Weng.json"
first_name = "Chunhua"
last_name = "Weng"

def preprocess_main(first_name, last_name, file_dir):
    author_dict = {
        "first_name": [],
        "last_name": [], 
        "num_publications": [],
        "mesh_term": [],
        "pmid": []
    }

    with open(file_dir) as json_f:
        output_f = json.load(json_f)

    num_pub = len(output_f)

    for o in output_f:
        authors = o.get("Authors", [])
        n_authors = len(authors)

        if n_authors == 0:
            continue

        # Compute ranges:
        first_three_end = min(3, n_authors)             
        last_three_start = max(0, n_authors - 3)        

        for loc, author in enumerate(authors):

            # name matching
            if last_name == author.get("Last Name", "") and first_name in author.get("First Name", ""):


                if loc < first_three_end or loc >= last_three_start:

                    for term in o.get("MeSH terms", []):
                        author_dict["first_name"].append(first_name)
                        author_dict["last_name"].append(last_name)
                        author_dict["num_publications"].append(num_pub)
                        author_dict["mesh_term"].append(term)
                        author_dict["pmid"].append(o.get("PMID"))


    return pd.DataFrame(author_dict)
