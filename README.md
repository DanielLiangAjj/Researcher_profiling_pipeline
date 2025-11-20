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

The pipeline operates in three main stages:

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

### 1\. Configure API Keys

Set your OpenAI API key in your environment variables:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2\. Generate a Profile

You can generate a profile by providing a researcher's name and their PubMed publication list.



## Citation

If you use this code or methodology in your research, please cite the original paper:

```
https://www.sciencedirect.com/science/article/abs/pii/S1532046425001789
```



## Acknowledgments

This work was supported by the National Center for Advancing Translational Sciences (NCATS) and the National Library of Medicine (NLM)
