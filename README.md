# doNt-Forget-your-Language
The official repository of our EACL2024-Findings paper: Understanding and Mitigating Spurious Correlations in Text Classification with Neighborhood Analysis.
* 🎯 Text classification models are prone to over reliance on spurious correlations that exist in the training set but may not hold true in general circumstances.
* ✅ We demystify this underlying representation space and propose a simple yet effective mitigation method, NFL.
* 📝 https://aclanthology.org/2024.findings-eacl.68/
* 🙌 This project is a collaborative effort between National Taiwan University (NTU), University of California, Los Angeles (UCLA) and University of Illinois Urbana-Champaign (UIUC).

Installation
---
Assuming Miniconda is installed, run the following commands to set up the environment:
```bash
conda create -n nfl python=3.8
conda activate nfl
pip install -r requirements.txt
```

Usage
---
**Data**: The processed datasets are available at `data`. If you wish to create those biased/unbiased/filtered subsets for your custom datasets, please refer to `src/create_subsets.py`

**Models**: To reproduce the numbers in the paper, simply run `src/nfl.py` for NFL or `src/dfr.py` for the DFR baseline that we re-implemented.

Citation
---
Please consider citing our paper if you find our work helpful. Thank you!
```bibtex
@inproceedings{chew-etal-2024-understanding,
    title = "Understanding and Mitigating Spurious Correlations in Text Classification with Neighborhood Analysis",
    author = "Chew, Oscar  and
      Lin, Hsuan-Tien  and
      Chang, Kai-Wei  and
      Huang, Kuan-Hao",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.68",
    pages = "1013--1025",
}
```