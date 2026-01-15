# ERD Similarity and Grade Prediction Tool

This project focuses on the automated assessment of Entity-Relationship Diagrams (ERDs) by predicting grades based on structural and semantic similarity. By leveraging a variety of techniques—including natural language processing (NLP), graph theory, and machine learning—the tool identifies patterns in diagrammatic data to provide consistent and accurate evaluations against a training set of graded examples.

---

## Project Overview
The primary goal of this tool is to bridge the gap between theoretical ERD design and automated data modeling challenges. It utilizes K-Nearest Neighbors (KNN) regression to predict the grade of an unlabeled diagram by finding the most similar diagrams in a curated training dataset.

### Core Functionalities
* **Text Normalization**: Comprehensive preprocessing of entity and attribute labels, including camelCase splitting, special character removal, and standardization of common abbreviations (e.g., mapping "num" to "number").
* **Structural Analysis**: Extraction of graph-based features such as arity, cardinality, and entity-relationship connections.
* **Similarity-Based Prediction**: Implementation of a weighted KNN regression model that uses similarity thresholds to ensure high-confidence predictions.

---

## Methodologies and Approaches

The system implements four distinct approaches, ranging from traditional text-based models to sophisticated hybrid graph-text analysis.

### 1. TF-IDF + KNN (Textual Vectorization)
This approach treats the ERD as a "bag-of-words" by extracting all entity names, attributes, and relationship labels.
* **Vectorization**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) to weight the importance of terms across the collection.
* **Dimensionality Reduction**: Implements Latent Semantic Analysis (LSA) via Singular Value Decomposition (SVD) to capture latent concepts and reduce noise.
* **Prediction**: Employs KNN with hyperparameter tuning for neighbors ($K$), distance metrics (Cosine/Euclidean), and weighting schemes.

### 2. Word Embeddings + KNN (Semantic Similarity)
Moving beyond keyword matching, this approach focuses on the semantic meaning of diagram labels.
* **Embeddings**: Utilizes models like Word2Vec or SBERT to generate dense vector representations of normalized tokens.
* **Semantic Averaging**: Represents each document by the mean vector of its constituent tokens, allowing the model to recognize semantic relationships between different terms.

### 3. Graph2Vec + Text Hybrid
This approach combines structural graph topology with textual features to create a robust hybrid similarity model.
* **Structural Embedding**: Uses **Graph2Vec** to convert the ERD graph (nodes representing entities, relationships, and attributes) into a single fixed-length vector.
* **The Hybrid Model**: A tunable hyperparameter ($A$) balances the weight between the structural Graph2Vec similarity and textual similarity (TF-IDF or Embeddings).
    $$Similarity = A \cdot Graph2vec\_similarity + (1 - A) \cdot Text\_similarity$$

### 4. Custom Feature Engineering (Graph + Text)
A granular, component-based approach that calculates similarity by explicitly comparing sub-components of the ERD.
* **Entity & Relationship Matching**: Uses greedy matching to pair entities and relationships between two diagrams based on specific attributes.
* **Component Similarity**: Calculates scores based on:
    * **Type Similarity**: Matching entity kinds (e.g., strong vs. weak entities).
    * **Name & Attribute Overlap**: Comparison of entity names and attribute lists using embeddings or Jaccard similarity.
    * **Structural Constraints**: Comparing arity (number of involved entities) and cardinalities (max/min constraints).
* **ERD Feature Vector**: Incorporates high-level features like the total count of entities, weak entities, and various relationship types.

---

## Performance Evaluation
Predictions are evaluated using **Root Mean Squared Error (RMSE)**. To ensure robustness and account for potential manual grading inconsistencies, the evaluation process excludes the worst 15% of predictions.

---

## Tech Stack
* **Language**: Python
* **Data Analysis**: Pandas, NumPy, SciPy
* **Machine Learning**: Scikit-learn (KNN Regressor, Cross-Validation)
* **NLP**: NLTK, Gensim (Word2Vec), Sentence-Transformers (SBERT)
* **Graph Processing**: NetworkX

---

## Authors
**Neel Vachhani, Shrung Patel, Matthew Sigit, Alex Liu**
