# Scalable Scientific Interest Profiling Using LLMs

## Overview

This repository contains the official implementation of the pipeline described in the paper **"Scalable Scientific Interest Profiling Using Large Language Models"** (Journal of Biomedical Informatics, 2025).

The pipeline automatically generates narrative research profiles for scientists by analyzing their publications and MeSH terms using GPT-4o.

## Key Features

* **Smart Filtering:** Keeps publications where the researcher is a primary contributor (first/last 3 author positions)
* **MeSH-based Profiling:** Expands MeSH terms through the hierarchy and uses TF-IDF to identify distinctive research themes
* **Dual Summaries:** Generates separate profiles for Health Domain focus and Methodological contributions

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, OpenAI API Key

## Quick Start

### Step 1: Preprocess Publications

```bash
python 01_filter_publications.py "data/input_sample/Chunhua Weng.json" "Chunhua Weng"
```

This filters publications and outputs `data/output_sample/preprocess/filtered_publications.csv`.

### Step 2: Run the Pipeline

1. Open `02_generate_research_profiles.ipynb`
2. Enter your OpenAI API key in the third cell
3. Run all cells

### Step 3: View Results

Final output: `results/intermediate_result/Research_summary_byMesh.xlsx`

## Input Format

Prepare a JSON file with publications, such a metadata and data format can be directly fetched via PubMed API (see `data/input_sample/Chunhua Weng.json`):

```json
[
    {
        "PMID": "12345678",
        "Title": "Paper Title",
        "Abstract": "Paper Abstract.",
        "Keywords": [
            "keyword1",
            "keyword2"
        ],
        "MeSH terms": [
            "MeSH Term1",
            "MeSH Term2"
        ],
        "Authors": [
            {
                "First Name": "John",
                "Last Name": "Doe",
                "Affiliation": "Institute Name"
            }
        ],
        "Journal": "Journal Name",
        "PubDate": "2026"
    },
```

## Project Structure

```
├── 01_filter_publications.py          # Step 1: Filter publications by author position
├── 02_generate_research_profiles.ipynb # Step 2: Main pipeline
├── mesh_tree_hierarchy.bin            # MeSH tree hierarchy
├── mesh_category_classification.xlsx  # Health/Methods classification
├── data/                              # Input/output samples
└── results/intermediate_result/       # Pipeline outputs
```

## Citation

```bibtex
@article{Liang2025Scalable,
  title   = {Scalable scientific interest profiling using large language models},
  author  = {Liang, Y. and Zhang, G. and Sun, E. and Idnay, B. and Fang, Y. and Chen, F. and Ta, C. and Peng, Y. and Weng, C.},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  volume  = {172},
  pages   = {104949},
  doi     = {10.1016/j.jbi.2025.104949}
}
```

## Acknowledgments

This work was supported by the National Center for Advancing Translational Sciences (NCATS) and the National Library of Medicine (NLM).
