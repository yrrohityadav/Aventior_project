# 📄 AI/DS Assignment — OCR-Based Batch Record Analysis

> **Automated extraction and comparison of pharmaceutical Batch Record data from scanned PDFs using OCR.**

---

## 📌 Problem Statement

Two scanned pharmaceutical Batch Record PDFs — **`Sample-9690.pdf`** and **`Sample-9700.pdf`** — contain structured manufacturing data as **image-only pages** (no embedded text layer). The goal is to extract, parse, and compare specific data points across both documents using **Optical Character Recognition (OCR)**.

### Questions Addressed

| # | Question | Summary |
|---|----------|---------|
| **Q1** | Batch Record Approval Signatures | Extract approver names, designations, companies, and dates from Page 1 of both PDFs; compute time differences |
| **Q3** | Equipment List Comparison | Compare equipment numbers (GC-MS, DSC, Dissolution Tester) across both PDFs and flag differences |
| **Q5** | Granulation: Equipment & Parameter Comparison | Determine if the same granulation equipment was used; compute changes in Mixing Time and Granulation Time |
| **Q6** | Milling: Data Extraction & Handwritten Note | Extract milling parameters from both PDFs and capture the handwritten annotation in Sample-9700 |

---

## 🛠️ Technology Stack

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3.x** | Core programming language |
| **Tesseract OCR** (v5.5+) | Optical character recognition engine |
| **PyMuPDF (fitz)** | PDF → high-resolution image rendering |
| **Pillow (PIL)** | Image pre-processing (grayscale, binarization) |
| **pytesseract** | Python wrapper for Tesseract |
| **Jupyter Notebook** | Interactive development and documentation |

---

## 🔬 Approach & Methodology

### OCR Pipeline

```
PDF Page ─▶ High-DPI Pixmap (300 DPI) ─▶ Grayscale ─▶ Binarize (threshold=180) ─▶ Tesseract OCR ─▶ Raw Text
```

### Key Design Decisions

- **DPI = 300**: Optimal balance between image quality and processing speed. A 400 DPI fallback is used for difficult pages.
- **Binary Threshold = 180**: Tuned specifically for dark text on light background, common in pharmaceutical batch records.
- **PSM 6 (Tesseract)**: Assumes a single uniform block of text — works well for full-page scanned documents.
- **Hybrid Approach**: Automated OCR extraction combined with manual verification ensures data integrity, especially for handwritten and stamped fields.

### Challenges Addressed

| Challenge | Solution |
|-----------|----------|
| Scanned images with no text layer | Full OCR pipeline with image pre-processing |
| Handwritten signatures and dates | Manual correction layer after initial OCR pass |
| Stamped approval marks overlapping text | Image binarization to isolate dark text |
| Low-contrast equipment numbers | Adaptive thresholding with fallback to higher DPI |
| Noisy scan artifacts | Grayscale conversion + binary thresholding to eliminate background noise |

---

## 📊 Key Results

### Question 1 — Approval Signatures

| Approver | Sample-9690 | Sample-9700 | Time Gap |
|----------|-------------|-------------|----------|
| **Tyrion Lannister** | 21-Aug-2023 | 23-Aug-2028 | ~5 years, 4 days |
| **Sandor Clegane** | 22-Aug-2023 | 28-Aug-2028 | ~5 years, 8 days |

> Sample-9700 was approved approximately **5 years** after Sample-9690, consistent with different manufacturing batches of the same drug product.

### Question 3 — Equipment Comparison

| Equipment | No. (Sample-9690) | No. (Sample-9700) | Same? |
|-----------|--------------------|--------------------|-------|
| GC-MS | GCMS-A71 | GCMS-A71 | ✅ Yes |
| DSC | DSC-05 | DSC-05 | ✅ Yes |
| Dissolution Tester | DT-009 | DT-009 | ✅ Yes |

> All three equipment numbers are **identical** — the same instruments were used across both batches.

### Question 5 — Granulation Parameters

| Parameter | Sample-9690 | Sample-9700 | Change |
|-----------|-------------|-------------|--------|
| **Equipment Used** | GK-004 | GK-003 | ❌ Different (Flag: No) |
| Mixing Time | 50 sec | 65 sec | **+15 sec** |
| Granulation Time | 300 sec | 400 sec | **+100 sec** |

> Different granulation equipment was used; both time parameters **increased** in Sample-9700.

### Question 6 — Milling & Handwritten Note

| Parameter | Sample-9690 (Page 6) | Sample-9700 (Page 7) |
|-----------|----------------------|----------------------|
| Equipment Used | ML-095 | ML-035 |
| Milling Time | 15 min | 20 min |
| Particle Size Achieved | 0.4 mm | 0.4 mm |

**Handwritten Note (Sample-9700):**
> *"Extra milling time as higher binder volume caused delay to achieve desired particle size"*

---

## 📁 Project Structure

```
RohitYadav_Assignment/
│
├── AI_DS_Assignment.ipynb          # 📓 Main consolidated notebook (all questions)
├── Assessment_AI_DS_Jan_28_2026.pdf # 📋 Original assignment PDF
├── Sample-9690.pdf                 # 📄 Batch Record PDF 1
├── Sample-9700.pdf                 # 📄 Batch Record PDF 2
├── create_notebook.py              # 🔧 Script to generate the notebook
│
├── Question_1/                     # Q1: Approval Signatures
│   ├── Q1_Approval_Signatures.ipynb
│   └── results/
│       ├── q1_approval_signatures.json
│       ├── q1_approvers_sample_9690.csv
│       ├── q1_approvers_sample_9700.csv
│       └── q1_time_difference.csv
│
├── Question_3/                     # Q3: Equipment Comparison
│   ├── Q3_Equipment_Comparison.ipynb
│   └── results/
│       ├── q3_equipment_comparison.csv
│       └── q3_equipment_comparison.json
│
├── Question_5/                     # Q5: Granulation Parameters
│   ├── Q5_Granulation_Comparison.ipynb
│   └── results/
│       ├── q5_equipment_comparison.csv
│       ├── q5_granulation_comparison.json
│       └── q5_granulation_parameters.csv
│
├── Question_6/                     # Q6: Milling & Handwritten Note
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

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install pymupdf pytesseract Pillow jupyter

# 4. Run the main notebook
jupyter notebook AI_DS_Assignment.ipynb
```

> **Note**: The notebook auto-installs any missing packages in its first cell.

---

## 📝 Output Formats

All extracted data is saved in **dual formats** as per assignment requirements:

| Format | Purpose | Location |
|--------|---------|----------|
| **CSV** | Tabular format — easily opened in Excel, pandas, or any spreadsheet tool | `Question_X/results/*.csv` |
| **JSON** | Dictionary format — structured nested data ideal for programmatic access | `Question_X/results/*.json` |

---

## 👤 Author

**Rohit Yadav**

---

## 📜 License

This project is submitted as part of the **Aventior AI/DS Assessment (Jan 2026)**.
