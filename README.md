# Researcher_profiling_pipeline
-----

# Scalable Scientific Interest Profiling Using LLMs

## Overview

[cite\_start]This repository contains the official implementation of the pipeline described in the paper **"Scalable Scientific Interest Profiling Using Large Language Models"**[cite: 1404].

[cite\_start]Research profiles are essential for talent discovery and collaboration but are often outdated or incomplete on major platforms[cite: 1412, 1432]. This tool addresses this gap by providing an automated, scalable pipeline that generates narrative scientific interest profiles for researchers.

[cite\_start]By leveraging **GPT-4o-mini**, this system requires user input of researchers' names and revelant publications (titles, abstracts, and MeSH terms) as input to create up-to-date, comprehensive research summaries[cite: 1414, 1501].

## Key Features

  * [cite\_start]**Smart Filtering:** Prioritizes relevant work by filtering for publications from the last decade where the researcher is a primary contributor (first three or senior author)[cite: 1442, 1468].
  * **Dual Profiling Strategies:**
      * [cite\_start]**MeSH-based Profiling:** Generates summaries using Medical Subject Headings (MeSH), and MeSH Term tree climb to automatically fill up revelant terms, to circumvent context window limits and focus on key semantic concepts[cite: 1492, 1286].
      * [cite\_start]**Abstract-based Profiling:** Uses a "Divide-and-Conquer" approach combined with Latent Dirichlet Allocation (LDA) to summarize vast amounts of abstract text[cite: 1494, 1496].

## Methodology

[cite\_start]The pipeline operates in three main stages[cite: 1463]:

1.  **Data Collection:**
Requires user input of
      * researcher metadata (name, affiliation).
      * [cite\_start]Retrieves publication records (Titles, Abstracts, MeSH terms) via NIH Entrez E-utilities[cite: 1467].
      * [cite\_start]Filters for significant contributions (first/last 3 authors)[cite: 1468].

2.  **Profile Generation (LLM):**

      * [cite\_start]Utilizes **GPT-4o-mini** for text generation[cite: 1501].
      * [cite\_start]**Prompt Engineering:** Adopts specific personas (e.g., "Dean of a college") to ensure professional and consistent summaries[cite: 1511].
      * [cite\_start]**Handling Long Contexts:** For researchers with extensive bibliographies, LDA topic modeling groups publications before summarization to fit token limits[cite: 1496].

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

### 1\. Configure API Keys

Set your OpenAI API key in your environment variables:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2\. Generate a Profile

You can generate a profile by providing a researcher's name and their PubMed publication list.



## Citation

If you use this code or methodology in your research, please cite the original paper:

```bibtex
@article{liang2025scalable,
  title={Scalable Scientific Interest Profiling Using Large Language Models},
  author={Liang, Yilun and Zhang, Gongbo and Sun, Edward and Idnay, Betina and Fang, Yilu and Chen, Fangyi and Talat, Casey and Peng, Yifan and Weng, Chunhua},
  journal={arXiv preprint arXiv:2508.15834},
  year={2025},
  url={https://arxiv.org/abs/2508.15834}
}
```

[cite\_start][cite: 1403, 1405]


## Acknowledgments

[cite\_start]This work was supported by the National Center for Advancing Translational Sciences (NCATS) and the National Library of Medicine (NLM)[cite: 1339, 1340].
