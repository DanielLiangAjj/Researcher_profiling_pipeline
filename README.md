# Scalable Scientific Interest Profiling Using LLMs

## Overview

This repository contains the official implementation of the pipeline described in the paper **"Scalable Scientific Interest Profiling Using Large Language Models"**.

Research profiles are essential for talent discovery and collaboration but are often outdated or incomplete on major platforms. This tool addresses this gap by providing an automated, scalable pipeline that generates narrative scientific interest profiles for researchers.

By leveraging **GPT-4o-mini**, this system requires user input of researchers' names and revelant publications (titles, abstracts, and MeSH terms) as input to create up-to-date, comprehensive research summaries.

## Key Features

  * **Smart Filtering:** Prioritizes relevant work by filtering for publications from the last decade where the researcher is a primary contributor (first three or senior author).
  * **Dual Profiling Strategies:**
      * **MeSH-based Profiling:** Generates summaries using Medical Subject Headings (MeSH), and MeSH Term tree climb to automatically fill up revelant terms, to circumvent context window limits and focus on key semantic concepts.
      * **Abstract-based Profiling:** Uses a "Divide-and-Conquer" approach combined with Latent Dirichlet Allocation (LDA) to summarize vast amounts of abstract text.

## Methodology

The pipeline operates in two main stages:

1.  **Data Collection:**
Requires user input of
      * researcher metadata (name, affiliation).
      * Retrieves publication records (Titles, Abstracts, MeSH terms) via NIH Entrez E-utilities.
      * Filters for significant contributions (first/last 3 authors).

2.  **Profile Generation (LLM):**

      * Utilizes **GPT-4o-mini** for text generation.
      * **Prompt Engineering:** Adopts specific personas (e.g., "Dean of a college") to ensure professional and consistent summaries.
      * **Handling Long Contexts:** For researchers with extensive bibliographies, LDA topic modeling groups publications before summarization to fit token limits.

## Installation

### Prerequisites

  * Python 3.8+
  * OpenAI API Key
  * PubMed API Key (optional but recommended for higher rate limits)

### Dependencies

Install the required packages:

```bash
pip install openai numpy scikit-learn nltk torch bert-score
```

## Usage

Please provide three input fields as a key-value dictionary into the pipeline:

1. OpenAI API Key
2. Researcher Name
3. Publication Record (a key-value pair, please see the data/input_sample/Chunhua Weng.json for the detailed format)




## Citation

If you use this code or methodology in your research, please cite the original paper:

```
@article{Liang2025Scalable,
  title   = {Scalable scientific interest profiling using large language models},
  author  = {Liang, Y. and Zhang, G. and Sun, E. and Idnay, B. and Fang, Y. and Chen, F. and Ta, C. and Peng, Y. and Weng, C.},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  month   = dec,
  volume  = {172},
  pages   = {104949},
  doi     = {10.1016/j.jbi.2025.104949},
  pmid    = {41177243},
  pmcid   = {PMC12705189}
}

```



## Acknowledgments

This work was supported by the National Center for Advancing Translational Sciences (NCATS) and the National Library of Medicine (NLM)
