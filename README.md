# Predicting Molecular Binding Affinities On The BELKA Dataset

## Overview
This repository contains the code and resources for our project on predicting molecular binding affinities. The project uses a novel hybrid feature engineering approach that integrates SMILES representations, ECFP fingerprints, word embeddings, Retrieval-Augmentation-Generation (RAG) methods, and vector search techniques to enhance the predictive power of machine learning models in determining molecular binding affinities.

## Motivation
Predicting small molecule binding affinity to protein targets is crucial for efficient and accurate drug discovery. Traditional and unimodal molecular representation methods often fail to capture the complex, multimodal nature of the data, as well as the rich contextual and structural information from molecular interactions necessary for accurate predictions. To address these challenges, we developed a comprehensive pipeline that leverages various feature representation techniques to achieve a rich, multimodal feature set.

## Methods

### Feature Engineering Techniques
- **SMILES Representations:** Provides a standardized textual format notation for molecules, capturing the basic structural information.
- **ECFP Fingerprints:** Encodes molecular structures into fixed-length binary vectors, capturing substructural information and facilitating efficient similarity comparisons.
- **MolFormer Word Embeddings:** Uses pretrained chemical language model (MolFormer) to encode SMILES sequences into high-dimensional vectors that capture semantic and structural relationships between molecules, enhancing contextual information.
- **Retrieval-Augmented Representation Learning:** Applies FAISS-based nearest-neighbor search over MolFormer embeddings to retrieve similar molecules, using attention-weighted aggregation of neighbor features to enrich original molecular represtnations, improving predictive signal.
- **Vector Search Techniques:** AUses FAISS for efficient approximate nearest neighbor search in high-dimensional embedding space, enabling rapid similarity-based feature enrichment across large molecular datasets.

### Modeling Approach
An ensemble learning strategy, combining Random Forest and XGBoost classifiers, is used to make predictions. This approach leverages the diverse strengths of each model, resulting in enhanced robustness and accuracy.

## Results
Preliminary evaluations indicate that this hybrid approach significantly improves model performance in predicting molecular bindings. Additionally, RAG helps to address data limitations by retrieving and generating relevant information to enrich the dataset.

## Getting the Data
The data for this project is sourced from the Kaggle competition "NeurIPS 2024 - Predict New Medicines with BELKA." You can access the competition and download the dataset from the following link:
[Kaggle Competition: NeurIPS 2024 - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA/data)

## Overview from Kaggle
In this competition, you’ll develop machine learning (ML) models to predict the binding affinity of small molecules to specific protein targets – a critical step in drug development for the pharmaceutical industry that would pave the way for more accurate drug discovery. You’ll help predict which drug-like small molecules (chemicals) will bind to three possible protein targets.

### Additional Details
- **Chemical Representations:** One of the goals of this competition is to explore and compare many different ways of representing molecules. Small molecules have been represented with SMILES, graphs, 3D structures, and more, including more esoteric methods such as spherical convolutional neural nets. Competitors are encouraged to explore not only different methods of making predictions but also to try different ways of representing the molecules.
- **SMILES:** SMILES is a concise string notation used to represent the structure of chemical molecules. It encodes the molecular graph, including atoms, bonds, connectivity, and stereochemistry as a linear sequence of characters, by traversing the molecule graph. SMILES is widely used in machine learning applications for chemistry, such as molecular property prediction, drug discovery, and materials design, as it provides a standardized and machine-readable format for representing and manipulating chemical structures.

## Downloading the Data
1. Sign up for a Kaggle account if you don't have one.
2. Go to the Kaggle competition page.
3. Accept the competition rules.
4. Download the dataset files and place them in the appropriate directory in your project.

## Repository Contents
- belka-ensemble-1-0-1.ipynb: Jupyter notebook containing the competition submission code for the entire pipeline, including data preprocessing, feature engineering, model training, and evaluation.


## Getting Started
### Prerequisites
Ensure you have the following packages installed:

- rdkit
- dgl
- duckdb
- faiss-gpu (or faiss-cpu)
- scikit-learn
- xgboost
- transformers
- torch
- pandas
- numpy

You can install these packages using pip:
```bash
pip install rdkit-pypi dgl duckdb faiss-gpu scikit-learn xgboost transformers torch pandas numpy
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/park-jsdev/belka-molecular-prediction.git
cd belka-molecular-prediction
```

2. Open the Jupyter notebook:
```bash
jupyter notebook belka-ensemble-1-0-1.ipynb
```

3. Follow the steps in the notebook to preprocess the data, engineer features, train the models, and evaluate the results.

### Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Acknowledgements

We would like to thank Leash Bio and Kaggle for making the dataset available, the contributors and the open-source community for their valuable resources and tools that made this project possible.
