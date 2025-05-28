# CLEF-2025-Checkthat-Task1-and-Task2

Welcome to our implementation for the CheckThat! Lab at CLEF 2025! This repository contains our solutions for Task 1 (Subjectivity) and Task 2 (Claims Extraction & Normalization). All our implementations are in Jupyter notebooks, making it easy to run on Google Colab or any Jupyter notebook environment.

## Our Team

### Students
- **Syed Muhammad Ather** (sh07554@st.habib.edu.pk)
- **Sidra Aamir** (sa07316@st.habib.edu.pk)
- **Muhammad Anas** (ma08458@st.habib.edu.pk)
- **Turab Usmani** (tu08125@st.habib.edu.pk)

### Faculty Advisors
- **Dr. Faisal Alvi** (faisal.alvi@sse.habib.edu.pk)
- **Dr. Abdul Samad** (abdul.samad@sse.habib.edu.pk)

### Affiliation
Department of Computer Science, Dhanani School of Science and Engineering, Habib University, Karachi, Pakistan

## Project Overview

### Task 1: Subjectivity
In this task, we tackle the challenge of distinguishing between subjective and objective content in news articles. Our system analyzes sentences and paragraphs to determine whether they express the author's personal views or present factual information. This binary classification task helps in understanding the nature of news content and its potential bias.

### Task 2: Claims Extraction & Normalization
Social media is full of complex, noisy content that can be hard to fact-check. Our solution simplifies these posts into clear, normalized claims, making it easier to verify information. We support 20 languages, making our system versatile and widely applicable.

## Project Structure

```
.
├── Task 01/          # Subjectivity Classification Implementation
│   ├── task1_monolingual_llm.ipynb        # Monolingual LLM implementation
│   ├── task1_multilingual_llm.ipynb       # Multilingual LLM implementation
│   ├── arabic_and_bulgarian_data_augmentation.py  # Data augmentation for Arabic and Bulgarian
│   └── english_data_augmentation.py       # Data augmentation for English
│
└── Task 02/          # Claims Extraction & Normalization Implementation
    ├── Task2_bartbase_spanish_korean_zeroshot.ipynb    # BART model for Spanish and Korean
    ├── Task2_checkthat-flan-t5_eng_spa.ipynb           # FLAN-T5 model for English and Spanish
    └── Task2_llama-3-2-1b-checkthat-english.ipynb      # LLaMA model for English
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- Required Python packages (listed in requirements.txt)

### Running the Notebooks
1. Clone this repository
2. Open the notebooks in Google Colab or Jupyter Notebook
3. Follow the instructions in each notebook
4. Make sure to have the required datasets in the data/ directory

## Implementation Details

### Task 1: Subjectivity Classification
Our implementation includes:
- Monolingual LLM approach for single-language classification
- Multilingual LLM approach for cross-language classification
- Data augmentation techniques for:
  - English text
  - Arabic and Bulgarian text
- State-of-the-art transformer models for classification

### Task 2: Claims Extraction & Normalization
We've implemented multiple approaches:
- BART-based model for Spanish and Korean zero-shot learning
- FLAN-T5 model for English and Spanish claim extraction
- LLaMA 3.2 1B model for English claim extraction
- Advanced NLP techniques for claim normalization

## Evaluation Results

### Task 1
- Accuracy: [To be added after evaluation]
- F1 Score: [To be added after evaluation]
- Precision: [To be added after evaluation]
- Recall: [To be added after evaluation]

### Task 2
- METEOR Score: [To be added after evaluation]
- Language-specific results will be added after evaluation


## Acknowledgments

We would like to express our gratitude to:
- The CheckThat! Lab organizers for providing the datasets and evaluation framework
- Our faculty advisors for their guidance and support
- The open-source community for the tools and libraries that made this project possible

## License

This project is licensed under the MIT License - see the LICENSE file for details.