# ERD Similarity and Grade Prediction Tool

This project focuses on the automated assessment of Entity-Relationship Diagrams (ERDs) by predicting grades based on structural and semantic similarity. [cite_start]By leveraging a variety of techniques—including natural language processing (NLP), graph theory, and machine learning—the tool identifies patterns in diagrammatic data to provide consistent and accurate evaluations against a training set of graded examples[cite: 2, 6, 7].

---

## Project Overview
[cite_start]The primary goal of this tool is to bridge the gap between theoretical ERD design and automated data modeling challenges[cite: 7]. [cite_start]It utilizes K-Nearest Neighbors (KNN) regression to predict the grade of an unlabeled diagram by finding the most similar diagrams in a curated training dataset[cite: 10, 11, 13].

### Core Functionalities
* [cite_start]**Text Normalization**: Comprehensive preprocessing of entity and attribute labels, including camelCase splitting, special character removal, and standardization of common abbreviations (e.g., mapping "num" to "number")[cite: 17, 20, 21].
* [cite_start]**Structural Analysis**: Extraction of graph-based features such as arity, cardinality, and entity-relationship connections[cite: 72, 76, 78, 94].
* [cite_start]**Similarity-Based Prediction**: Implementation of a weighted KNN regression model that uses similarity thresholds to ensure high-confidence predictions[cite: 14, 15, 16].

---

## Methodologies and Approaches

[cite_start]The system implements four distinct approaches, ranging from traditional text-based models to sophisticated hybrid graph-text analysis[cite: 30].

### 1. TF-IDF + KNN (Textual Vectorization)
[cite_start]This approach treats the ERD as a "bag-of-words" by extracting all entity names, attributes, and relationship labels[cite: 32].
* [cite_start]**Vectorization**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) to weight the importance of terms across the collection[cite: 6].
* **Dimensionality Reduction**: Implements Latent Semantic Analysis (LSA) via Singular Value Decomposition (SVD) to capture latent concepts and reduce noise.
* [cite_start]**Prediction**: Employs KNN with hyperparameter tuning for neighbors ($K$), distance metrics (Cosine/Euclidean), and weighting schemes[cite: 6, 11].

### 2. Word Embeddings + KNN (Semantic Similarity)
[cite_start]Moving beyond keyword matching, this approach focuses on the semantic meaning of diagram labels[cite: 33].
* [cite_start]**Embeddings**: Utilizes models like Word2Vec or SBERT to generate dense vector representations of normalized tokens[cite: 6, 23, 56].
* [cite_start]**Semantic Averaging**: Represents each document by the mean vector of its constituent tokens, allowing the model to recognize semantic relationships between different terms[cite: 50, 75].

### 3. Graph2Vec + Text Hybrid
[cite_start]This approach combines structural graph topology with textual features to create a robust hybrid similarity model[cite: 34, 38].
* [cite_start]**Structural Embedding**: Uses **Graph2Vec** to convert the ERD graph (nodes representing entities, relationships, and attributes) into a single fixed-length vector[cite: 34, 36, 37].
* [cite_start]**The Hybrid Model**: A tunable hyperparameter ($A$) balances the weight between the structural Graph2Vec similarity and textual similarity (TF-IDF or Embeddings)[cite: 38, 39, 41, 42].
    $$Similarity = A \cdot Graph2vec\_similarity + (1 - A) \cdot Text\_similarity$$

### 4. Custom Feature Engineering (Graph + Text)
[cite_start]A granular, component-based approach that calculates similarity by explicitly comparing sub-components of the ERD[cite: 44].
* [cite_start]**Entity & Relationship Matching**: Uses greedy matching to pair entities and relationships between two diagrams based on specific attributes[cite: 60, 90].
* **Component Similarity**: Calculates scores based on:
    * [cite_start]**Type Similarity**: Matching entity kinds (e.g., strong vs. weak entities)[cite: 46, 52, 53].
    * [cite_start]**Name & Attribute Overlap**: Comparison of entity names and attribute lists using embeddings or Jaccard similarity[cite: 47, 49, 50].
    * [cite_start]**Structural Constraints**: Comparing arity (number of involved entities) and cardinalities (max/min constraints)[cite: 72, 76, 78, 80, 83, 84].
* [cite_start]**ERD Feature Vector**: Incorporates high-level features like the total count of entities, weak entities, and various relationship types[cite: 93, 94, 96].

---

## Performance Evaluation
[cite_start]Predictions are evaluated using **Root Mean Squared Error (RMSE)**[cite: 8]. [cite_start]To ensure robustness and account for potential manual grading inconsistencies, the evaluation process excludes the worst 15% of predictions[cite: 8, 9].

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
