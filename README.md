# Autism Detection
## Project Description
This repository contains the implementation code for the paper:  
**Towards the Development of Explainable Machine Learning Models to Recognize the Faces of Autistic Children**

**This project leverages machine learning techniques to detect autism spectrum disorder (ASD) from various data sources. The goal is to build accurate and interpretable models that can assist in early diagnosis.**

You can find the paper [here](https://www.emerald.com/aia/article-abstract/11/4/283/1271080/Towards-the-development-of-explainable-machine?redirectedFrom=fulltext)

---

## Features

- Data preprocessing and exploration
- Model training and evaluation
- Visualization of results
- Easy-to-use scripts and notebooks

## 🧪 How to Run the Testing Pipeline
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Autism-Detection.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the best weight from [here](https://drive.google.com/drive/folders/1aYxVtVVZX6XV9tELmNhsSHEYg2t8zbCC?usp=sharing)

**Dataset**

The dataset can be accessed from [here](https://www.kaggle.com/discussions/general/123978)

**Test the model**

To run the test script:
```bash
python test.py
```

**Train the model**

To run the train script, you need to first change the values in the config.py, and then run the following script:
```bash
python train.py
```

**Use Explainability**

To run the explainability, you need to run the following script:
```bash
python explainability.py
```

## Project Structure

```bash
Autism-Detection/
├── utils/               # Utility modules (audio, data, general helpers)
│   ├── __init__.py
│   ├── data_utils.py
│   ├── general.py
│   ├── model_utils.py
│   ├── training.py
│   └── xai_utils.py
├── config.py            # Configuration Settings
├── constants.py         # Global constants
├── explainability.py    # Model explainability and interpretation tools
├── test.py              # Testing script
├── train.py             # Training script
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License information
```

## License

This project is licensed under the **[MIT License](LICENSE)**.

## Disclaimer

This tool is for research and educational purposes only. It is not intended for clinical use.


## Citation
If you use this code, please cite:
```bibtex
@article{10.1108/AIA-02-2025-0018,
    author = {Omrani, Ali Reza and Lanovaz, Marc J. and Moroni, Davide},
    title = {Towards the development of explainable machine learning models to recognize the faces of autistic children: a brief report},
    journal = {Advances in Autism},
    volume = {11},
    number = {4},
    pages = {283-289},
    year = {2025},
    month = {08},    
    issn = {2056-3868},
    doi = {10.1108/AIA-02-2025-0018},
    url = {https://doi.org/10.1108/AIA-02-2025-0018},
    eprint = {https://www.emerald.com/aia/article-pdf/11/4/283/10047266/aia-02-2025-0018en.pdf},
}
