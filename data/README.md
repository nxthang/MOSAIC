# MOSAIC - Datasets

Directory containing datasets used for the MOSAIC project.

## 📊 Overview

| Dataset | Size | Number of Samples | Status |
|---------|------|-------------------|--------|
| **CulturePark** | 7.2 MB | 41,000+ | ✅ Downloaded |
| **NORMAD** | 46 MB | 2,630+ | ⚠️ Requires decryption |

---

## 1. CulturePark Dataset

**Reference:** Wang et al. (2024). CulturePark: Boosting Cross-cultural Understanding in Large Language Models. NeurIPS 2024.

**GitHub:** https://github.com/Scarelette/CulturePark  
**arXiv:** https://arxiv.org/abs/2405.15145

### 📁 Data Structure

```
data/culturepark/
├── data/
│   ├── Chinese/          # Chinese cultural data
│   ├── Arabic/           # Arabic cultural data
│   ├── German/           # German cultural data
│   ├── Korean/           # Korean cultural data
│   ├── Spanish/          # Spanish cultural data
│   ├── Portuguese/       # Portuguese cultural data
│   ├── Bengali/          # Bengali cultural data
│   ├── Turkish/          # Turkish cultural data
│   ├── WVQ.csv           # World Values Questionnaire
│   └── WVQ.jsonl         # WVQ JSONL format
├── data_process.py       # Data processing script
├── main.py               # Main generation script
└── README.md             # Documentation
```

### 📍 Cultural Regions (12 dimensions)

The dataset includes data from 8+ countries/regions, mapped according to Hofstede's cultural dimensions:

1. **Power Distance**
2. **Individualism vs Collectivism**
3. **Masculinity vs Femininity**
4. **Uncertainty Avoidance**
5. **Long-term Orientation**
6. **Indulgence vs Restraint**

### 🔧 Usage

```python
# Load CulturePark data
import pandas as pd

# Load WVQ data
wvq_df = pd.read_csv('data/culturepark/data/WVQ.csv')
wvq_jsonl = pd.read_json('data/culturepark/data/WVQ.jsonl', lines=True)

# Load cultural-specific data
china_df = pd.read_csv('data/culturepark/data/Chinese/China.csv')
germany_df = pd.read_csv('data/culturepark/data/Germany/Germany.csv')
```

---

## 2. NORMAD Dataset

**Reference:** Arora et al. (2024). NORMAD: Norms and Morals Across Dialects for Sociocultural Alignment Evaluation. arXiv:2404.12464

**GitHub:** https://github.com/Akhila-Yerukola/NormAd  
**HuggingFace:** https://huggingface.co/datasets/akhilayerukola/NormAd

### 📁 Data Structure

```
data/normad/
├── data_and_heval/
│   ├── datasets.zip.enc      # Main dataset (encrypted)
│   ├── human_eval_inhouse/   # Human evaluation (in-house)
│   └── human_eval_mturk/     # Human evaluation (MTurk)
├── story_prompts/            # Prompts for story generation
├── conf/                     # Configuration files
├── src/                      # Source code
└── README.md                 # Documentation
```

### ⚠️ Important Note

The main dataset (`datasets.zip.enc`) **is encrypted**. To access:

1. **Contact authors:** Email Akhila Yerukola (akhila@seas.upenn.edu) for decryption key
2. **Use HuggingFace:** Load directly from HuggingFace datasets:
   ```python
   from datasets import load_dataset
   normad = load_dataset("akhilayerukola/NormAd", split="train")
   ```
3. **Use GitHub data:** Some data samples are available in `story_prompts/` and `src/analysis/`

### 📊 Dataset Statistics

- **2,630+ samples** (train split on HuggingFace)
- **75 countries** covered
- **305 cultural backgrounds**
- **20 cultural sub-axes**

### 🔧 Usage (HuggingFace)

```python
from datasets import load_dataset

# Load from HuggingFace
normad = load_dataset("akhilayerukola/NormAd", split="train")

# View columns
print(normad.column_names)
# ['ID', 'Country', 'Background', 'Axis', 'Subaxis', 
#  'Value', 'Rule-of-Thumb', 'Story', 'Explanation', 'Gold Label']

# Sample
sample = normad[0]
print(f"Country: {sample['Country']}")
print(f"Story: {sample['Story']}")
print(f"Label: {sample['Gold Label']}")
```

---

## 3. Combining Datasets

To create a training dataset for Cultural Equilibrium:

```python
import pandas as pd
from pathlib import Path

# Load CulturePark
culturepark_data = []
for csv_file in Path('data/culturepark/data').rglob('*.csv'):
    df = pd.read_csv(csv_file)
    df['source'] = 'culturepark'
    culturepark_data.append(df)
culturepark_df = pd.concat(culturepark_data, ignore_index=True)

# Load NORMAD (from HuggingFace)
from datasets import load_dataset
normad = load_dataset("akhilayerukola/NormAd", split="train")
normad_df = normad.to_pandas()
normad_df['source'] = 'normad'

# Combine
combined_df = pd.concat([culturepark_df, normad_df], ignore_index=True)
print(f"Total samples: {len(combined_df)}")
```

---

## 📝 Download Instructions (for future users)

### CulturePark

```bash
cd data/culturepark
git clone https://github.com/Scarelette/CulturePark.git .
```

### NORMAD

**Option 1: From GitHub**
```bash
cd data/normad
git clone https://github.com/Akhila-Yerukola/NormAd.git .
# Note: datasets.zip.enc requires decryption key from authors
```

**Option 2: From HuggingFace (recommended)**
```python
from datasets import load_dataset
normad = load_dataset("akhilayerukola/NormAd", split="train")
normad.to_json("data/normad/normad_train.json")
```

---

## 📊 Statistics (Summary)

| Metric | CulturePark | NORMAD | Total |
|--------|-------------|--------|-------|
| Total samples | 41,000+ | 2,630 | ~43,630 |
| Cultural regions | 8+ | 75 countries | 83+ |
| Languages | 8 | 1 (English) | 8 |
| Format | CSV, JSONL | JSON, CSV | - |
| License | MIT | CC-BY-4.0 | - |

---

## 🔗 Links

- **CulturePark:**
  - GitHub: https://github.com/Scarelette/CulturePark
  - Paper: https://arxiv.org/abs/2405.15145
  - NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2024/hash/77f089cd16dbc36ddd1caeb18446fbdd-Abstract-Conference.html

- **NORMAD:**
  - GitHub: https://github.com/Akhila-Yerukola/NormAd
  - HuggingFace: https://huggingface.co/datasets/akhilayerukola/NormAd
  - Paper: https://arxiv.org/abs/2404.12464

---

## 📞 Contact

**Author:** Thang Nguyen Xuan  
**Institution:** Hanoi University, Vietnam  


**Last Updated:** 2026-05-11
