# Highly Trustworthy Multiomics Learning (HTML)
Accurate cancer diagnosis and prognosis are crucial for improving medical care delivery. Multiomics data offers a comprehensive cellular view of cancer, but is often leveraged with static models of late-fusion integration and infers certain cancer. Here, we present HTML (Highly Trustworthy Multiomics Learning), a dynamic multimodal integration method, and show how to apply HTML to integrate multiomics data for pan-cancer and cancer-subtype diagnosis and prognosis. Compared with static models which have fixed computational graphs and parameters at inference, HTML can adapt structures to each input, leading to notable advantages of individualized analysis. We show HTML filters out sample-adaptive features, aligns modality-wise representations, produces interpretable predictions, and identifies important biomarkers. Comprehensive assessments on a 33-type pan-cancer dataset and 12 cancer-subtype datasets reveal HTML's superiority over other state-of-the-art approaches by a large margin. These results suggest that HTML is an effective tool for personalizing cancer therapeutic strategies and biological pathogenesis discovery.

## Getting Started

To get start with HTML, please follow the instructions below.

### Clone the repository

```
git clone git@github.com:YuxingLu613/HTML.git
```

### Prerequisites

```
pip install -r requirements.txt
```
### To run the main script

```
python main.py
```

or using bash script

```
bash run.sh
```

You can use the provided scripts to preprocess data, train models, and evaluate results. Here's a brief overview of the main scripts:

- **preprocessing.py**: This script handles data preprocessing, including normalization and preparation for k-fold cross-validation.
- **train.py**: This script trains models using k-fold cross-validation and evaluates their performance.
- **main.py**: This is the main entry point of the project, orchestrating data preprocessing, model training, and evaluation.


### Dataset
The BRCA and ROSMAP dataset can be found in [MOGONET](https://github.com/txWang/MOGONET). All the preprocessed datasets can be found in [HTML Datasets](https://drive.google.com/drive/folders/1_tJ2ekhTmWp7ZcRVjUVGx0cqGMRKEhNo?usp=share_link). The pan-cancer dataset can be fetched in [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) program.

## Contribution
If you wish to contribute to this project, please submit a pull request or open an issue on GitHub. All contributions are welcome!

## Citation
If you utilize the resources in this repository and need to cite them, please refer to the following publication:

```
Yuxing Lu, Rui Peng, Lingkai Dong, Kun Xia, Renjie Wu, Shuai Xu, Jinzhuo Wang.
"Multiomics dynamic learning enables personalized diagnosis and prognosis for pancancer and cancer subtypes."
Briefings in Bioinformatics, Volume 24, Issue 6, November 2023.
[Link](https://doi.org/10.1093/bib/bbad378).
```

## Contact
If you have any questions, please feel free to get touch with me, my email is yxlu0613 AT gmail DOT com
