"""
Research Profile Generation Pipeline

This script processes MeSH terms from researcher publications and generates
natural language research profiles using GPT-4.

Usage:
    python 02_generate_research_profiles.py --api-key YOUR_API_KEY --input filtered_publications.csv

Example:
    python 02_generate_research_profiles.py \\
        --api-key sk-xxx \\
        --input data/output_sample/preprocess/filtered_publications.csv \\
        --output results/intermediate_result
"""

import argparse
import os
import sys
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from kneed import KneeLocator
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

MEANINGLESS_MESH_TERMS = [
    'Eukaryota', 'Animals', 'Chordata', 'Vertebrates', 'Mammals', 'Eutheria',
    'Primates', 'Haplorhini', 'Catarrhini', 'Hominidae', 'Humans',
    'Natural Science Disciplines', 'Science', 'Research', 'Methods',
    'Investigative Techniques', 'Persons', 'Health Occupations',
    'Equipment and Supplies', 'Electrical Equipment and Supplies',
    'Biomedical Research', 'Household Products', 'Photography',
    'Financial Management', 'life', 'Medicine', 'Diseases'
]

DEFAULT_MESH_TREE_FILE = "data/reference_files/mesh_tree_hierarchy.bin"
DEFAULT_CLASS_FILE = "data/reference_files/mesh_category_classification.xlsx"
DEFAULT_OUTPUT_DIR = "results/final_result"


# =============================================================================
# MESH PROCESSING FUNCTIONS
# =============================================================================

def load_mesh_trees(meshtree_file):
    """Load MeSH tree file and build ID-to-name and name-to-ID mappings."""
    mesh_id2name = {}
    mesh_name2id = defaultdict(list)

    with open(meshtree_file, "r") as ftree:
        for line in ftree:
            term, tree_id = line.strip().split(";")
            mesh_id2name[tree_id] = term
            mesh_name2id[term].append(tree_id)

    # Add synthetic entries for Female/Male
    mesh_id2name['NEWID1'] = 'Female'
    mesh_id2name['NEWID2'] = 'Male'
    mesh_name2id['Female'] = ['NEWID1']
    mesh_name2id['Male'] = ['NEWID2']

    return mesh_id2name, mesh_name2id


def process_mesh_terms(csv_file, mesh_id2name, mesh_name2id):
    """Process researcher MeSH terms into hierarchical MeSH ancestor codes."""

    def split_mesh_term(mesh_term):
        parts = mesh_term.split(" / ")
        return (parts[0], parts[1:]) if len(parts) > 1 else (parts[0], [])

    def find_mesh_term_ancestors(tid, ancestor_ids):
        ancestors = []
        mesh_name = mesh_id2name[tid]
        ancestors.append(mesh_name)

        ids_for_name = mesh_name2id[mesh_name]
        ancestor_ids += ids_for_name
        parent_ids = [".".join(mid.split(".")[:-1]) for mid in ids_for_name]

        for pid in parent_ids:
            if pid and pid not in ancestor_ids:
                new_ancestors, ancestor_ids = find_mesh_term_ancestors(pid, ancestor_ids)
                ancestors.extend(new_ancestors)
                ancestors = list(set(ancestors))
        return ancestors, ancestor_ids

    def create_mesh_term_hierarchical_codes(tid):
        ancestors, _ = find_mesh_term_ancestors(tid, [])
        return ancestors

    df_mesh = pd.read_csv(csv_file)
    df_mesh['mesh_term_only'], df_mesh['mesh_subheading'] = zip(
        *df_mesh['MeSH Term'].apply(split_mesh_term)
    )
    df_mesh['mesh_term_only_id'] = df_mesh['mesh_term_only'].apply(lambda x: mesh_name2id[x])
    df_mesh.loc[df_mesh['mesh_term_only'] == 'Female', 'mesh_term_only_id'] = 'NEWID1'
    df_mesh.loc[df_mesh['mesh_term_only'] == 'Male', 'mesh_term_only_id'] = 'NEWID2'
    df_mesh = df_mesh.explode("mesh_term_only_id")

    sys.setrecursionlimit(5000)
    df_mesh['ancestor_mesh_term'] = df_mesh['mesh_term_only_id'].apply(create_mesh_term_hierarchical_codes)
    df_mesh = df_mesh.explode('ancestor_mesh_term')
    df_mesh = df_mesh[['First Name', 'Last Name', 'PMID', 'Person ID', 'ancestor_mesh_term']]
    df_mesh.drop_duplicates(inplace=True)

    return df_mesh


def filter_low_frequency_mesh_terms(df_mesh_term, min_frequency=2):
    """Filter out MeSH terms occurring less than min_frequency times."""
    df_freq = (
        df_mesh_term
        .groupby('Person ID')['ancestor_mesh_term']
        .value_counts()
        .rename('count')
        .reset_index()
    )
    return df_freq[df_freq['count'] >= min_frequency]


def categorize_mesh_terms(df_mesh_term_freq, mesh_id2name, mesh_name2id, class_file):
    """Categorize MeSH terms into Health Domain and Method categories."""
    df_class = pd.read_excel(class_file, index_col=0)
    mesh_names_health_domain = list(df_class[df_class['Class'] == 'H']['name'])
    mesh_names_method = list(df_class[df_class['Class'] == 'M']['name'])

    df_temp = df_mesh_term_freq[['ancestor_mesh_term']].drop_duplicates()
    df_temp['term_id'] = df_temp['ancestor_mesh_term'].apply(lambda x: mesh_name2id[x])
    df_temp = df_temp.explode('term_id')
    df_temp['length'] = df_temp['term_id'].apply(len)
    df_temp = df_temp.sort_values(by='length').drop_duplicates()

    for tid in df_temp['term_id']:
        term_name = mesh_id2name[tid]
        if term_name in mesh_names_health_domain or term_name in mesh_names_method:
            continue
        parts = tid.split(".")
        if len(parts) >= 2:
            parent_id = ".".join(parts[:-1])
            parent_name = mesh_id2name[parent_id]
            if parent_name in mesh_names_health_domain:
                mesh_names_health_domain.append(term_name)
            if parent_name in mesh_names_method:
                mesh_names_method.append(term_name)

    return list(set(mesh_names_health_domain)), list(set(mesh_names_method))


def remove_meaningless_mesh_terms(df_mesh_term_freq, mesh_id2name):
    """Remove overly broad or non-informative MeSH terms."""
    terms_to_remove = MEANINGLESS_MESH_TERMS.copy()
    for mesh_id, name in mesh_id2name.items():
        if mesh_id.startswith('Z01'):
            terms_to_remove.append(name)
    return df_mesh_term_freq[~df_mesh_term_freq['ancestor_mesh_term'].isin(terms_to_remove)].copy()


# =============================================================================
# TF-IDF FUNCTIONS
# =============================================================================

def export_mesh_term_frequency_by_category(df_mesh, df_mesh_term_freq, mesh_names_health_domain, mesh_names_method, directory):
    """Export MeSH term frequency tables by category."""
    df_person = df_mesh[['First Name', 'Last Name', 'Person ID']].drop_duplicates()

    df_health = df_mesh_term_freq[
        df_mesh_term_freq['ancestor_mesh_term'].isin(mesh_names_health_domain)
    ].merge(df_person, on='Person ID')

    df_method = df_mesh_term_freq[
        df_mesh_term_freq['ancestor_mesh_term'].isin(mesh_names_method)
    ].merge(df_person, on='Person ID')

    os.makedirs(directory, exist_ok=True)
    df_health.to_excel(os.path.join(directory, "mesh_term_freq_per_faculty_HealthDomain.xlsx"), index=False)
    df_method.to_excel(os.path.join(directory, "mesh_term_freq_per_faculty_Method.xlsx"), index=False)

    return df_health.values.tolist(), df_method.values.tolist()


def build_researcher_mesh_string(mesh_term_freq_list):
    """Build frequency-weighted term strings for TF-IDF input."""
    dict_researcher_mesh = defaultdict(str)
    for row in mesh_term_freq_list:
        person_id, term, freq = row[:3]
        dict_researcher_mesh[person_id] += (term + ";") * int(freq)
    return {pid: terms.rstrip(";") for pid, terms in dict_researcher_mesh.items()}


def run_mesh_tfidf(dict_researcher_mesh, df_person, directory, postfix="HealthDomain"):
    """Compute TF-IDF scores for researcher MeSH term profiles."""
    def custom_tokenizer(text):
        return [token.strip() for token in text.split(';') if token.strip()]

    corpus = list(dict_researcher_mesh.values())
    faculty_ids = list(dict_researcher_mesh.keys())

    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
    X = vectorizer.fit_transform(corpus)

    df_tfidf = pd.DataFrame(X.toarray())
    df_tfidf["Person ID"] = faculty_ids
    df_tfidf.set_index("Person ID", inplace=True)
    df_tfidf = df_tfidf.T
    df_tfidf["mesh_term"] = vectorizer.get_feature_names_out()
    df_tfidf.set_index("mesh_term", inplace=True)
    df_tfidf = pd.DataFrame(df_tfidf.unstack()).reset_index()
    df_tfidf.rename(columns={"level_0": "Person ID", 0: "tfidf_score"}, inplace=True)
    df_tfidf = df_tfidf.merge(df_person, on="Person ID")
    df_tfidf["name"] = df_tfidf["First Name"] + " " + df_tfidf["Last Name"]
    df_tfidf.drop(["Person ID", "First Name", "Last Name"], axis=1, inplace=True)

    os.makedirs(directory, exist_ok=True)
    df_tfidf.to_csv(os.path.join(directory, f"term_per_researcher_tfidf_{postfix}.csv"), index=False)

    return df_tfidf


# =============================================================================
# GPT-4 SUMMARIZATION FUNCTIONS
# =============================================================================

def create_gpt4_response_generator(api_key):
    """Create a GPT-4 response generator function with the given API key."""
    client = OpenAI(api_key=api_key)

    def generate_gpt4_response(content, print_output=False):
        try:
            completions = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                top_p=0.1,
                n=1,
                messages=[
                    {'role': 'system', 'content': 'You are a dean of a college.'},
                    {'role': 'user', 'content': content},
                ]
            )
            if print_output:
                print(completions)
            return completions.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    return generate_gpt4_response


def generate_research_focus_summaries(postfix, directory, generate_gpt4_response, df_person, elbow_S=5):
    """Generate research focus summaries for each researcher using GPT-4."""
    file_path = os.path.join(directory, f"term_per_researcher_tfidf_{postfix}.csv")
    df_tfidf = pd.read_csv(file_path)
    df_tfidf = df_tfidf[df_tfidf["tfidf_score"] > 0]

    dict_mesh_tfidf = (
        df_tfidf.groupby("name")
        .apply(lambda x: dict(zip(x["mesh_term"], x["tfidf_score"])))
        .to_dict()
    )

    dict_terms = {}
    dict_summaries = {}

    for researcher_name in tqdm(dict_mesh_tfidf.keys(), desc=f"Generating {postfix} summaries"):
        df_temp = df_tfidf[df_tfidf["name"] == researcher_name].copy()
        df_temp.sort_values("tfidf_score", ascending=False, inplace=True)
        df_temp["rank"] = np.arange(len(df_temp))

        kneedle = KneeLocator(
            df_temp["rank"], df_temp["tfidf_score"],
            S=elbow_S, curve="convex", direction="decreasing"
        )
        knee_point = kneedle.knee if kneedle.knee else max(3, int(0.05 * len(df_temp)))

        selected_terms = list(df_temp.iloc[:knee_point]["mesh_term"])
        dict_terms[researcher_name] = "; ".join(selected_terms)

        prompt = (
            f"Help me summarize this group of phrases into 1 sentence as a research focus:\n"
            f"{dict_terms[researcher_name]}\n"
            f"Please start with: The research focus is on"
        )

        summary = None
        while summary is None:
            summary = generate_gpt4_response(prompt)
            if summary is None:
                print("GPT-4 failed, retrying...")
                time.sleep(3)

        if postfix == "Method":
            summary = summary.replace("The research focus is on", "This researcher has mainly contributed to")
        else:
            summary = summary.replace("The research focus is on", "This researcher mainly focused on")

        dict_summaries[researcher_name] = summary

    with open(os.path.join(directory, f"dict_terms_for_a_researcher_for_focus{postfix}.pickle"), "wb") as f:
        pickle.dump(dict_terms, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(directory, f"dict_research_focus_for_a_researcher_{postfix}.pickle"), "wb") as f:
        pickle.dump(dict_summaries, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_terms, dict_summaries


def combine_research_focus_summaries(directory, output_excel="Research_summary_byMesh.xlsx"):
    """Combine Health Domain and Method summaries into a single research profile."""
    with open(os.path.join(directory, "dict_research_focus_for_a_researcher_HealthDomain.pickle"), "rb") as f:
        dict_hd = pickle.load(f)
    df_hd = pd.DataFrame.from_dict(dict_hd, orient="index").reset_index()
    df_hd.columns = ["Researcher_name", "Research_summary_hd"]

    with open(os.path.join(directory, "dict_research_focus_for_a_researcher_Method.pickle"), "rb") as f:
        dict_m = pickle.load(f)
    df_m = pd.DataFrame.from_dict(dict_m, orient="index").reset_index()
    df_m.columns = ["Researcher_name", "Research_summary_m"]

    df = df_hd.merge(df_m, on="Researcher_name", how="outer").fillna("")
    df["Research_direction"] = (df["Research_summary_hd"] + "\n" + df["Research_summary_m"]).str.strip("\n")
    df_final = df[["Researcher_name", "Research_direction"]]
    df_final.to_excel(os.path.join(directory, output_excel), index=False)

    return df_final


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(api_key, input_file, output_dir, mesh_tree_file, class_file):
    """
    Run the complete research profile generation pipeline.

    Parameters
    ----------
    api_key : str
        OpenAI API key (required).
    input_file : str
        Path to filtered publications CSV from step 1.
    output_dir : str
        Directory for output files.
    mesh_tree_file : str
        Path to MeSH tree hierarchy file.
    class_file : str
        Path to MeSH category classification file.
    """
    print("=" * 60)
    print("Research Profile Generation Pipeline")
    print("=" * 60)

    # Step 1: Load MeSH tree
    print("\n[Step 1/10] Loading MeSH tree hierarchy...")
    mesh_id2name, mesh_name2id = load_mesh_trees(mesh_tree_file)
    print(f"  Loaded {len(mesh_id2name)} MeSH terms")

    # Step 2: Process MeSH terms
    print("\n[Step 2/10] Processing MeSH terms and expanding to ancestors...")
    df_mesh_term = process_mesh_terms(input_file, mesh_id2name, mesh_name2id)
    print(f"  Processed {len(df_mesh_term)} term-publication pairs")

    # Step 3: Filter low-frequency terms
    print("\n[Step 3/10] Filtering low-frequency terms...")
    df_mesh_term_freq = filter_low_frequency_mesh_terms(df_mesh_term, min_frequency=2)
    print(f"  Kept {len(df_mesh_term_freq)} term-researcher pairs")

    # Step 4: Categorize terms
    print("\n[Step 4/10] Categorizing terms into Health Domain vs Methods...")
    mesh_names_health_domain, mesh_names_method = categorize_mesh_terms(
        df_mesh_term_freq, mesh_id2name, mesh_name2id, class_file
    )
    print(f"  Health Domain terms: {len(mesh_names_health_domain)}")
    print(f"  Method terms: {len(mesh_names_method)}")

    # Step 5: Remove meaningless terms
    print("\n[Step 5/10] Removing meaningless/overly broad terms...")
    df_mesh_term_freq = remove_meaningless_mesh_terms(df_mesh_term_freq, mesh_id2name)
    print(f"  Remaining terms: {len(df_mesh_term_freq)}")

    # Step 6: Export frequencies by category
    print("\n[Step 6/10] Exporting term frequencies by category...")
    mesh_term_freq_list_hd, mesh_term_freq_list_m = export_mesh_term_frequency_by_category(
        df_mesh_term, df_mesh_term_freq, mesh_names_health_domain, mesh_names_method, output_dir
    )
    print(f"  Health Domain entries: {len(mesh_term_freq_list_hd)}")
    print(f"  Method entries: {len(mesh_term_freq_list_m)}")

    # Step 7: Build frequency-weighted strings
    print("\n[Step 7/10] Building frequency-weighted term strings...")
    dict_researcher_mesh_hd = build_researcher_mesh_string(mesh_term_freq_list_hd)
    dict_researcher_mesh_m = build_researcher_mesh_string(mesh_term_freq_list_m)

    # Step 8: Compute TF-IDF
    print("\n[Step 8/10] Computing TF-IDF scores...")
    df_person = df_mesh_term[['First Name', 'Last Name', 'Person ID']].drop_duplicates()
    run_mesh_tfidf(dict_researcher_mesh_hd, df_person, output_dir, postfix="HealthDomain")
    run_mesh_tfidf(dict_researcher_mesh_m, df_person, output_dir, postfix="Method")
    print("  TF-IDF computation complete")

    # Step 9: Generate GPT-4 summaries
    print("\n[Step 9/10] Generating research focus summaries with GPT-4...")
    generate_gpt4_response = create_gpt4_response_generator(api_key)
    generate_research_focus_summaries("Method", output_dir, generate_gpt4_response, df_person)
    generate_research_focus_summaries("HealthDomain", output_dir, generate_gpt4_response, df_person)

    # Step 10: Combine summaries
    print("\n[Step 10/10] Combining summaries and exporting final results...")
    df_summary = combine_research_focus_summaries(output_dir)
    print(f"  Generated profiles for {len(df_summary)} researchers")

    output_path = os.path.join(output_dir, "Research_summary_byMesh.xlsx")
    print("\n" + "=" * 60)
    print(f"Pipeline complete! Output saved to: {output_path}")
    print("=" * 60)

    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate research profiles from MeSH terms using GPT-4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_generate_research_profiles.py --api-key sk-xxx --input filtered_publications.csv
  python 02_generate_research_profiles.py --api-key sk-xxx --input data.csv --output results/
        """
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API key (required)"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/output_sample/preprocess/filtered_publications.csv",
        help="Path to filtered publications CSV (default: data/output_sample/preprocess/filtered_publications.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--mesh-tree",
        default=DEFAULT_MESH_TREE_FILE,
        help=f"Path to MeSH tree file (default: {DEFAULT_MESH_TREE_FILE})"
    )
    parser.add_argument(
        "--class-file",
        default=DEFAULT_CLASS_FILE,
        help=f"Path to category classification file (default: {DEFAULT_CLASS_FILE})"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.mesh_tree):
        print(f"Error: MeSH tree file not found: {args.mesh_tree}")
        sys.exit(1)

    if not os.path.exists(args.class_file):
        print(f"Error: Classification file not found: {args.class_file}")
        sys.exit(1)

    run_pipeline(
        api_key=args.api_key,
        input_file=args.input,
        output_dir=args.output,
        mesh_tree_file=args.mesh_tree,
        class_file=args.class_file
    )


if __name__ == "__main__":
    main()
