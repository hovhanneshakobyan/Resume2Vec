# Resume2Vec — ATS Resume Optimizer

# Generative AI
**Course:** Generative AI  
**Lecturer:** V. Avetisyan  
**University:** NPUA  

**Students:**  
- Hovhannes Hakobyan  
- Hakobyan Spartak


Resume2Vec is a Generative AI project for optimizing resumes for Applicant Tracking Systems (ATS).

The system:
- parses resumes from PDF, DOCX, or text
- compares them against a job description
- computes an ATS score
- identifies missing mandatory and optional requirements
- uses a fine-tuned semantic model for deeper matching
- generates an ATS-oriented optimized draft
- shows before/after comparison and text diff

---

## Overview

Traditional ATS systems often rely heavily on exact keyword matching. Resume2Vec improves on this by combining:

- structured resume parsing
- fine-tuned semantic similarity
- mandatory vs optional requirement weighting
- ATS scoring
- safe optimization and explanation

This allows the system to handle both:
- exact matches (`ASP.NET` ↔ `ASP.NET`)
- related matches (`MAUI` ↔ mobile / Android-related work)

---

## Architecture

```text
Resume (PDF / DOCX / text) + Job Description
                    │
                    ▼
              [ Resume Parser ]
                    │
                    ▼
          [ Section Extraction Layer ]
     - skills
     - education
     - experience
     - formatting indicators
                    │
                    ▼
      [ Fine-Tuned Semantic Matcher ]
   - trained on resume dataset
   - category-aware / self-supervised similarity
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 [ ATS Keyword / Skill Match ]  [ Semantic Scoring ]
        │                       │
        └───────────┬───────────┘
                    ▼
             [ ATS Scoring Engine ]
     - keyword score
     - skill score
     - semantic score
     - section quality score
     - format score
     - achievement score
                    │
                    ▼
          [ Resume Optimization Layer ]
     - professional summary generation
     - role alignment block
     - missing requirement suggestions
     - ATS-oriented draft generation
                    │
                    ▼
           [ Before / After Comparison ]
     - original score
     - optimized score
     - suggestions
     - text diff

Main Features
Resume parsing from PDF, DOCX, and text
Fine-tuned semantic resume matcher
ATS score from 0–100
Mandatory vs optional requirement handling
Skill extraction and related-skill reasoning
Format-risk detection for ATS readability
Before/after optimized draft generation
Resume diff visualization in Streamlit
Models
Component	Purpose	Base
Fine-Tuned Semantic Matcher	Resume ↔ JD semantic similarity	all-MiniLM-L6-v2
ATS Scoring Engine	Multi-factor ATS scoring	Custom logic
Resume Optimizer	ATS-oriented draft generation	Safe rule-based optimization
Training Strategy

The project includes a fine-tuned semantic model.

Semantic Model

The semantic matcher is trained using resume data placed in data/raw/.

Depending on dataset format, training can use:

category-aware supervision if labels exist
or self-supervised pair generation if only raw resume files are available

Examples of training pairs:

positive pairs: chunks from the same resume
negative pairs: chunks from different resumes

This allows the model to learn domain-specific semantic similarity for resume content.

Scoring Dimensions

The ATS score is built from multiple components:

Keyword score — how well resume terms align with JD terms
Skill score — direct and related skill coverage
Semantic score — deeper similarity between JD requirements and resume content
Section score — presence and quality of important sections
Format score — ATS readability and layout quality
Achievement score — quality of responsibility and impact evidence
Requirement Priorities

Job description requirements are split into:

mandatory
optional
contextual

Missing mandatory requirements reduce the score more than missing optional items.

Project Structure
GEN AI proj/
├── app.py
├── train_all.py
├── evaluate.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── resumes.csv
│   │   └── *.docx
│   └── processed/
│       ├── semantic_pairs.csv
│       └── label_map.json
├── models/
│   ├── siamese_checkpoint.pt
│   ├── optimizer_t5/
│   ├── generator_t5/
│   └── semantic_resume_matcher/
└── src/
    ├── parser.py
    ├── sections.py
    ├── jd_rules.py
    ├── data_prep.py
    ├── semantic_model.py
    ├── scorer.py
    ├── optimizer.py
    ├── diff_utils.py
    ├── config.py
    └── utils.py
Installation
pip install -r requirements.txt
Dataset Setup

Place your training data inside:

data/raw/

Supported formats:

resumes.csv
multiple .docx resumes directly inside data/raw/

No subfolders are required if using DOCX files.

Training

Run the full training pipeline:

python train_all.py

This performs:

data preparation
training pair generation
fine-tuning of the semantic similarity model
Evaluation

Run:

python evaluate.py

This evaluates:

semantic matching performance
ATS score behavior
optional text similarity metrics
Launch the App
streamlit run app.py
How It Works
User uploads a resume
User pastes a job description
The resume is parsed and structured
Skills, education, and experience evidence are extracted
The fine-tuned semantic model compares resume content with JD requirements
ATS score is computed
Missing mandatory and optional requirements are identified
An ATS-oriented optimized draft is generated
The app displays before/after score, suggestions, and diff
Example Output

The app provides:

original ATS score
optimized ATS score
matched skills
missing mandatory requirements
missing optional requirements
semantic report
ATS format warnings
optimized resume draft
before/after diff
Limitations
If a resume has no real experience bullets, responsibility scoring is limited
Highly visual or multi-column resumes may parse poorly for ATS analysis
Generated optimized drafts are ATS-oriented and should still be reviewed manually
The optimizer improves presentation and alignment, but does not invent factual experience
Future Improvements
stronger role-title extraction
better experience bullet reconstruction
LLM-based safe bullet generation
more advanced mandatory/optional classification
recruiter-style evaluation set
multilingual resume and JD support
References
Sentence Transformers / all-MiniLM-L6-v2
ATS resume optimization workflows
Resume semantic similarity methods
weakly supervised representation learning for document matching
