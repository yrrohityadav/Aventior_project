# 📊 Project Report — Aventior AI/DS Assessment (Jan 2026)

**Submitted by:** Rohit Yadav  
**Date:** February 2026  
**Repository:** [github.com/yrrohityadav/Aventior_project](https://github.com/yrrohityadav/Aventior_project)

---

## 📌 Executive Summary

This report documents the complete solution to the **Aventior AI/DS Assessment** (January 28, 2026). The assignment required automated extraction, parsing, and comparison of pharmaceutical **Batch Record** data from two scanned, image-only PDF documents — `Sample-9690.pdf` and `Sample-9700.pdf` — using **Optical Character Recognition (OCR)**.

A robust OCR pipeline was designed and implemented in Python using **Tesseract OCR**, **PyMuPDF**, and **Pillow**. Each of the four questions addressed has its own dedicated Jupyter Notebook with structured code, commentary, and output files in CSV and JSON formats.

---

## 🗂️ Assignment Overview

| Question | Topic | Difficulty |
|----------|-------|------------|
| **Q1** | Batch Record Approval Signatures | ⭐⭐⭐ |
| **Q3** | Equipment List Comparison | ⭐⭐ |
| **Q5** | Granulation: Equipment & Parameters | ⭐⭐⭐ |
| **Q6** | Milling Data & Handwritten Annotation | ⭐⭐⭐⭐ |

**Source Documents:**
- `Sample-9690.pdf` — Pharmaceutical Batch Record, Batch 9690 (approved Aug 2023)
- `Sample-9700.pdf` — Pharmaceutical Batch Record, Batch 9700 (approved Aug 2028)

---

## 🛠️ Technology Stack

| Tool / Library | Version | Purpose |
|----------------|---------|---------|
| Python | 3.8+ | Core programming language |
| Tesseract OCR | v5.5+ | Optical character recognition engine |
| PyMuPDF (`fitz`) | Latest | PDF → high-resolution image rendering |
| Pillow (PIL) | Latest | Image preprocessing (grayscale, binarization) |
| pytesseract | Latest | Python wrapper for Tesseract |
| Jupyter Notebook | Latest | Interactive development & documentation |
| pandas | Latest | Data manipulation and CSV export |

---

## 🔬 OCR Methodology

### Pipeline Architecture

```
PDF Page
  │
  ▼
PyMuPDF (300 DPI Pixmap Rendering)
  │
  ▼
PIL Image Conversion
  │
  ▼
Grayscale Conversion
  │
  ▼
Binary Thresholding (threshold = 180)
  │
  ▼
Tesseract OCR (PSM 6 — Uniform Block)
  │
  ▼
Raw Extracted Text
  │
  ▼
Regex / Keyword Parsing
  │
  ▼
Structured Data (DataFrame / Dictionary)
  │
  ├──▶ CSV Output
  └──▶ JSON Output
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **DPI = 300** | Optimal balance between OCR accuracy and processing speed; 400 DPI fallback for difficult pages |
| **Binary Threshold = 180** | Tuned for dark text on light background, standard in pharmaceutical batch records |
| **PSM 6** | Assumes a single uniform block of text — ideal for full-page scanned documents |
| **Hybrid Approach** | Automated OCR + manual correction layer for handwritten/stamped fields |
| **Fallback Mechanism** | If primary OCR extraction fails, regex fallback patterns search raw text |

### Challenges & Solutions

| Challenge | Solution Applied |
|-----------|-----------------|
| Image-only PDFs (no embedded text) | Full OCR pipeline with image preprocessing |
| Handwritten signatures and dates | Manual correction layer after initial OCR pass |
| Stamped approval marks overlapping text | Image binarization to isolate dark ink |
| Low-contrast equipment numbers | Adaptive thresholding with high-DPI fallback (400 DPI) |
| Noisy scan artifacts | Grayscale + binary thresholding to eliminate background noise |
| Inconsistent field formatting | Regex patterns with multiple fallback alternatives |

---

## 📋 Question 1 — Batch Record Approval Signatures

### Objective
Extract the **approver names**, **designations**, **companies**, and **approval dates** from **Page 1** of both PDF batch records. Compute the **time difference** between corresponding approvals.

### Notebook
📓 [`Question_1/Q1_Approval_Signatures.ipynb`](Question_1/Q1_Approval_Signatures.ipynb)

### Approach
1. Rendered Page 1 of each PDF at 300 DPI using PyMuPDF
2. Converted to grayscale and applied binary threshold (180)
3. Ran Tesseract OCR with `--psm 6` configuration
4. Parsed approver table using regex patterns to match name, designation, company, and date fields
5. Computed date differences using Python's `datetime` module
6. Exported results to CSV and JSON

### Results

**Sample-9690 — Approvers:**

| Approver Name | Designation | Company | Approval Date |
|---------------|-------------|---------|---------------|
| Tyrion Lannister | QA Manager | Pharma Co. A | 21-Aug-2023 |
| Sandor Clegane | Production Head | Pharma Co. A | 22-Aug-2023 |

**Sample-9700 — Approvers:**

| Approver Name | Designation | Company | Approval Date |
|---------------|-------------|---------|---------------|
| Tyrion Lannister | QA Manager | Pharma Co. A | 23-Aug-2028 |
| Sandor Clegane | Production Head | Pharma Co. A | 28-Aug-2028 |

**Time Differences:**

| Approver | Sample-9690 Date | Sample-9700 Date | Time Gap |
|----------|-----------------|-----------------|----------|
| Tyrion Lannister | 21-Aug-2023 | 23-Aug-2028 | **~5 years, 2 days** |
| Sandor Clegane | 22-Aug-2023 | 28-Aug-2028 | **~5 years, 6 days** |

### Key Finding
> Sample-9700 was approved approximately **5 years** after Sample-9690, consistent with two different manufacturing batches of the same pharmaceutical product across separate production cycles.

### Output Files
| File | Format | Description |
|------|--------|-------------|
| [`q1_approvers_sample_9690.csv`](Question_1/results/q1_approvers_sample_9690.csv) | CSV | Approver data for Batch 9690 |
| [`q1_approvers_sample_9700.csv`](Question_1/results/q1_approvers_sample_9700.csv) | CSV | Approver data for Batch 9700 |
| [`q1_time_difference.csv`](Question_1/results/q1_time_difference.csv) | CSV | Time difference comparison |
| [`q1_approval_signatures.json`](Question_1/results/q1_approval_signatures.json) | JSON | Complete structured result |

---

## 📋 Question 3 — Equipment List Comparison

### Objective
Extract and **compare equipment numbers** for three specific instruments — **GC-MS**, **DSC**, and **Dissolution Tester** — from both batch records. Flag any differences.

### Notebook
📓 [`Question_3/Q3_Equipment_Comparison.ipynb`](Question_3/Q3_Equipment_Comparison.ipynb)

### Approach
1. Identified the equipment table page in each PDF
2. Rendered at 300 DPI and applied standard OCR preprocessing
3. Used targeted keyword search (e.g., "GC-MS", "DSC", "Dissolution") to locate equipment rows
4. Extracted equipment numbers using regex patterns (e.g., `[A-Z]+-\d+`)
5. Compared extracted numbers side-by-side and flagged discrepancies

### Results

| Equipment | Batch 9690 No. | Batch 9700 No. | Match? |
|-----------|----------------|----------------|--------|
| GC-MS | GCMS-A71 | GCMS-A71 | ✅ Same |
| DSC | DSC-05 | DSC-05 | ✅ Same |
| Dissolution Tester | DT-009 | DT-009 | ✅ Same |

### Key Finding
> All three analytical instruments used **identical equipment numbers** across both batches, confirming the same instruments were employed for testing both `Sample-9690` and `Sample-9700`.

### Output Files
| File | Format | Description |
|------|--------|-------------|
| [`q3_equipment_comparison.csv`](Question_3/results/q3_equipment_comparison.csv) | CSV | Side-by-side equipment comparison |
| [`q3_equipment_comparison.json`](Question_3/results/q3_equipment_comparison.json) | JSON | Structured comparison with match flags |

---

## 📋 Question 5 — Granulation: Equipment & Parameter Comparison

### Objective
Determine whether the **same granulation equipment** was used across both batches. Extract and compute changes in **Mixing Time** and **Granulation Time** parameters.

### Notebook
📓 [`Question_5/Q5_Granulation_Comparison.ipynb`](Question_5/Q5_Granulation_Comparison.ipynb)

### Approach
1. Located the Granulation section in each PDF (searched across multiple pages)
2. Applied OCR at 300 DPI with binary thresholding
3. Extracted equipment ID using regex patterns matching equipment code formats
4. Parsed Mixing Time and Granulation Time values (numeric + unit extraction)
5. Computed absolute and percentage changes between batches

### Results

**Equipment Comparison:**

| Parameter | Sample-9690 | Sample-9700 | Same Equipment? |
|-----------|-------------|-------------|-----------------|
| Granulation Equipment | GK-004 | GK-003 | ❌ **No — Different** |

**Process Parameter Changes:**

| Parameter | Sample-9690 | Sample-9700 | Change | % Change |
|-----------|-------------|-------------|--------|----------|
| Mixing Time | 50 sec | 65 sec | **+15 sec** | +30% |
| Granulation Time | 300 sec | 400 sec | **+100 sec** | +33.3% |

### Key Findings
> 1. **Different granulation equipment** was used — GK-004 (Batch 9690) vs. GK-003 (Batch 9700).
> 2. **Mixing Time increased by 30%** (50s → 65s) in Batch 9700.
> 3. **Granulation Time increased by 33.3%** (300s → 400s) in Batch 9700, suggesting process modifications were made between batches.

### Output Files
| File | Format | Description |
|------|--------|-------------|
| [`q5_equipment_comparison.csv`](Question_5/results/q5_equipment_comparison.csv) | CSV | Equipment ID comparison |
| [`q5_granulation_parameters.csv`](Question_5/results/q5_granulation_parameters.csv) | CSV | Time parameter comparison |
| [`q5_granulation_comparison.json`](Question_5/results/q5_granulation_comparison.json) | JSON | Complete structured result |

---

## 📋 Question 6 — Milling Data & Handwritten Annotation

### Objective
Extract **milling parameters** (equipment, milling time, particle size) from both batch records and capture the **handwritten annotation** present in `Sample-9700`.

### Notebook
📓 [`Question_6/Q6_Milling_Handwritten.ipynb`](Question_6/Q6_Milling_Handwritten.ipynb)

### Approach
1. Located milling section pages (Page 6 for Sample-9690, Page 7 for Sample-9700)
2. Rendered at 300 DPI with standard preprocessing; used 400 DPI second pass for handwritten text
3. Extracted equipment ID, milling time, and particle size using targeted regex
4. Applied enhanced OCR (inverted image + high DPI) to detect and read the handwritten note
5. Manual verification of handwritten text to ensure accuracy

### Results

**Milling Process Comparison:**

| Parameter | Sample-9690 (Page 6) | Sample-9700 (Page 7) | Change |
|-----------|---------------------|---------------------|--------|
| Equipment Used | ML-095 | ML-035 | ❌ Different |
| Milling Time | 15 min | 20 min | **+5 min (+33.3%)** |
| Particle Size Achieved | 0.4 mm | 0.4 mm | ✅ Same |

**Handwritten Note (Sample-9700 only):**

> *"Extra milling time as higher binder volume caused delay to achieve desired particle size"*

### Key Findings
> 1. **Different milling equipment** used (ML-095 vs. ML-035).
> 2. Despite using different equipment and 33% more milling time in Batch 9700, the **same particle size (0.4 mm)** was achieved.
> 3. The handwritten annotation explains the delay — **higher binder volume** in Batch 9700 required extended milling to reach target specifications.

### Output Files
| File | Format | Description |
|------|--------|-------------|
| [`q6_milling_comparison.csv`](Question_6/results/q6_milling_comparison.csv) | CSV | Milling parameter comparison |
| [`q6_handwritten_text.csv`](Question_6/results/q6_handwritten_text.csv) | CSV | Extracted handwritten annotation |
| [`q6_milling_and_handwritten.json`](Question_6/results/q6_milling_and_handwritten.json) | JSON | Complete structured result |

---

## 📁 Project Structure

```
RohitYadav_Assignment/
│
├── 📓 AI_DS_Assignment.ipynb           # Main consolidated notebook (all questions)
├── 📋 Assessment_AI_DS_Jan_28_2026.pdf # Original assignment PDF
├── 📄 Sample-9690.pdf                  # Batch Record — Batch 9690
├── 📄 Sample-9700.pdf                  # Batch Record — Batch 9700
├── 🔧 create_notebook.py               # Script to programmatically generate notebooks
├── 📊 REPORT.md                        # ← This document
├── 📖 README.md                        # Project overview & quick start
│
├── Question_1/                         # Q1: Approval Signatures
│   ├── Q1_Approval_Signatures.ipynb
│   └── results/
│       ├── q1_approval_signatures.json
│       ├── q1_approvers_sample_9690.csv
│       ├── q1_approvers_sample_9700.csv
│       └── q1_time_difference.csv
│
├── Question_3/                         # Q3: Equipment Comparison
│   ├── Q3_Equipment_Comparison.ipynb
│   └── results/
│       ├── q3_equipment_comparison.csv
│       └── q3_equipment_comparison.json
│
├── Question_5/                         # Q5: Granulation Parameters
│   ├── Q5_Granulation_Comparison.ipynb
│   └── results/
│       ├── q5_equipment_comparison.csv
│       ├── q5_granulation_comparison.json
│       └── q5_granulation_parameters.csv
│
├── Question_6/                         # Q6: Milling & Handwritten Note
│   ├── Q6_Milling_Handwritten.ipynb
│   └── results/
│       ├── q6_handwritten_text.csv
│       ├── q6_milling_and_handwritten.json
│       └── q6_milling_comparison.csv
│
└── .gitignore
```

---

## 🚀 How to Run

### Prerequisites

1. **Python 3.8+** installed
2. **Tesseract OCR** installed and available in PATH
   - Windows: [Download from UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

### Setup & Execution

```bash
# 1. Clone the repository
git clone https://github.com/yrrohityadav/Aventior_project.git
cd Aventior_project

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install pymupdf pytesseract Pillow jupyter pandas

# 4. Run the main consolidated notebook
jupyter notebook AI_DS_Assignment.ipynb

# OR run individual question notebooks
jupyter notebook Question_1/Q1_Approval_Signatures.ipynb
jupyter notebook Question_3/Q3_Equipment_Comparison.ipynb
jupyter notebook Question_5/Q5_Granulation_Comparison.ipynb
jupyter notebook Question_6/Q6_Milling_Handwritten.ipynb
```

> **Note:** The notebooks auto-install any missing packages in the first cell.

---

## 📊 Summary of All Results

| Question | Key Metric | Sample-9690 | Sample-9700 | Outcome |
|----------|-----------|-------------|-------------|---------|
| **Q1** | Tyrion Lannister Approval | 21-Aug-2023 | 23-Aug-2028 | ~5 year gap |
| **Q1** | Sandor Clegane Approval | 22-Aug-2023 | 28-Aug-2028 | ~5 year gap |
| **Q3** | GC-MS Equipment No. | GCMS-A71 | GCMS-A71 | ✅ Same |
| **Q3** | DSC Equipment No. | DSC-05 | DSC-05 | ✅ Same |
| **Q3** | Dissolution Tester No. | DT-009 | DT-009 | ✅ Same |
| **Q5** | Granulation Equipment | GK-004 | GK-003 | ❌ Different |
| **Q5** | Mixing Time | 50 sec | 65 sec | +15 sec (+30%) |
| **Q5** | Granulation Time | 300 sec | 400 sec | +100 sec (+33.3%) |
| **Q6** | Milling Equipment | ML-095 | ML-035 | ❌ Different |
| **Q6** | Milling Time | 15 min | 20 min | +5 min (+33.3%) |
| **Q6** | Particle Size | 0.4 mm | 0.4 mm | ✅ Same |
| **Q6** | Handwritten Note | — | *"Extra milling time..."* | Captured ✅ |

---

## 📝 Output Format Summary

All extracted data is saved in **two formats** as per assignment requirements:

| Format | Description | Location |
|--------|-------------|----------|
| **CSV** | Tabular format — readable in Excel, pandas, Google Sheets | `Question_X/results/*.csv` |
| **JSON** | Dictionary format — structured for programmatic access | `Question_X/results/*.json` |

---

## 👤 Author Information

| Field | Details |
|-------|---------|
| **Name** | Rohit Yadav |
| **GitHub** | [@yrrohityadav](https://github.com/yrrohityadav) |
| **Repository** | [Aventior_project](https://github.com/yrrohityadav/Aventior_project) |
| **Assessment** | Aventior AI/DS Assessment — January 28, 2026 |
| **Submission Date** | February 2026 |

---

## 📜 Declaration

I hereby declare that this assignment has been completed independently. The OCR pipeline, data extraction logic, and analysis were implemented entirely by me. All results have been verified for accuracy against the source PDF documents.

---

*Report generated as part of the Aventior AI/DS Technical Assessment submission.*
