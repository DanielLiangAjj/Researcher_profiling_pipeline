import json 
import os
import pandas as pd
import re

def preprocess_main(first_name, last_name, file_dir):
    author_dict = {
                "first_name": [],
                "last_name": [], 
                "num_publications": [],
                "mesh_term": [],
                "pmid":[]}
    with open(file_dir) as json_f:
        output_f = json.load(json_f)
        num_pub = len(output_f)
        author_mesh_term_list = []
         
    for o in output_f:
        for loc, author in enumerate(o["Authors"]):
                # check if author last name match with the defined researcher
            if (last_name ==  author['Last Name']):
                    # check if author first name match  (using partial matching given middle name can be omitted)
                if (first_name in author["First Name"]):
                    if len(o["Authors"]) <=3:
                        for term in o["MeSH terms"]:
                            author_dict["first_name"].append(first_name)
                            author_dict["last_name"].append(last_name)
                            author_dict["num_publications"].append(num_pub)
                            author_dict["mesh_term"].append(term)
                            author_dict["pmid"].append(o["PMID"])

                                
                        else:
                            if loc <= 2:
                                for term in o["MeSH terms"]:
                                    author_dict["first_name"].append(first_name)
                                    author_dict["last_name"].append(last_name)
                                    author_dict["num_publications"].append(num_pub)
                                    author_dict["mesh_term"].append(term)
                                    author_dict["pmid"].append(o["PMID"])



                            elif len(o["Authors"]) -3 <= loc <= len(o["Authors"]) -1:
                                for term in o["MeSH terms"]:
                                    author_dict["first_name"].append(first_name)
                                    author_dict["last_name"].append(last_name)
                                    author_dict["num_publications"].append(num_pub)
                                    author_dict["mesh_term"].append(term)
                                    author_dict["pmid"].append(o["PMID"])


                            else: 
                                continue   
                    else:
                        continue


                else:
                    continue
                
        
        
    return pd.DataFrame(author_dict)

if __name__=='__main__':
    file_dir  = "test_names/"
    first_name = 'Chunhua'
    last_name = 'Weng'
    df = preprocess_main(first_name, last_name, file_dir)
    df.to_csv('sample_output_part1.csv')