import os
import json
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import nltk
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize

# --- 1. Stage 2 Normalization Functions ---

def split_camel_case(name):
    # Splits camelCase, pascalcase words into separate words
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

def get_normalized_tokens(json_string, use_stemming=False):
    # Normalizes tokens without stemming
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        return []

    raw_terms = []
    for entity in data.get('entities', []):
        raw_terms.append(entity.get('name', ''))
        raw_terms.extend(entity.get('primary_keys', []))
        raw_terms.extend(entity.get('attributes', []))

    for relationship in data.get('relationships', []):
        raw_terms.extend(relationship.get('attributes', []))

    normalized_tokens = []
    
    for item in raw_terms:
        if not item:
            continue
        item = split_camel_case(item)
        item = item.lower()
        item = re.sub(r'[_\-\/\\]', ' ', item)
        item = re.sub(r'[^a-z\s]', '', item)
        item = re.sub(r'\s+', ' ', item).strip()
        item = item.replace('num', 'number').replace('id', 'identifier')
        item = item.replace('no', 'number') 
        tokens = item.split()
        if use_stemming: # Should be unreachable
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        normalized_tokens.extend(tokens)

    return normalized_tokens 

def get_document_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size if hasattr(model, 'vector_size') else 100)


def run_approach_2(train_jsons, test_jsons, erd_no, dataset1_grade, dataset2_grade, GLOBAL_MEAN_GRADE):

    print("\nRunning Approach 2")
    # Data Preprocessing and training word2vec
    all_jsons = {**train_jsons, **test_jsons}
    processed_docs_dict = {}
    for key, json_string in all_jsons.items():
        processed_docs_dict[key] = get_normalized_tokens(json_string, use_stemming=False)

    processed_docs = list(processed_docs_dict.values())
    model = Word2Vec(processed_docs, vector_size=100, min_count=1, window=5, workers=3)
    
    # create document vectors
    sorted_train_keys = sorted(train_jsons.keys())
    sorted_test_keys = sorted(test_jsons.keys())

    X_train = np.array([get_document_vector(processed_docs_dict[key], model) for key in sorted_train_keys])
    X_test = np.array([get_document_vector(processed_docs_dict[key], model) for key in sorted_test_keys])

    # Normalize
    if X_train.size > 0:
        X_train = normalize(X_train)
    if X_test.size > 0:
        X_test = normalize(X_test)


    # Tune KNN
    y_train = []
    for key in sorted_train_keys:
        erd_num, dataset_num = key
        idx = erd_no.index(erd_num)
        grade = dataset1_grade[idx] if dataset_num == 1 else dataset2_grade[idx]
        if pd.notna(grade):
             y_train.append(grade)
    y_train = np.array(y_train)

    if X_train.shape[0] == 0:
        print("Error: Training set is empty. Cannot train KNN.")
        return

    # Hyperparameter Tuning
    best_rmse = float('inf')
    best_k = 5
    best_weights = 'distance' 
    best_metric = 'cosine' 

    if X_train.shape[0] >= 5:
        for k in [3, 5, 7, 9, 11, 13]:
            knn = KNeighborsRegressor(n_neighbors=k, weights=best_weights, metric=best_metric)
            cv_rmse = np.sqrt(-cross_val_score(knn, X_train, y_train, cv=min(5, len(y_train)),
                                                scoring='neg_mean_squared_error').mean())
            if cv_rmse < best_rmse:
                best_rmse = cv_rmse
                best_k = k

    print(f"\nApproach 2 Tuning: K={best_k}, weights={best_weights}, metric={best_metric}, CV RMSE={best_rmse:.3f}")

    final_knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights, metric=best_metric)
    final_knn.fit(X_train, y_train)

    # Predictions with Stage 2 rules
    predictions = []
    q_threshold = 0.5 # Similarity threshold

    for i, x_vec in enumerate(X_test):
        if X_train.shape[0] < best_k: # If too little data
            predicted_grade = GLOBAL_MEAN_GRADE
        else:
            k_predict = max(best_k, 2)
            distances, indices = final_knn.kneighbors([x_vec], n_neighbors=k_predict)
            similarities = 1 - distances[0][:best_k]
            valid_indices = indices[0][:best_k][similarities >= q_threshold] # similarity > threshold

            if len(valid_indices) >= 1:
                valid_neighbors_grades = y_train[valid_indices]
                weights = similarities[similarities >= q_threshold]
                predicted_grade = np.average(valid_neighbors_grades, weights=weights)

            elif np.all(similarities < q_threshold):
                predicted_grade = GLOBAL_MEAN_GRADE
            else:
                predicted_grade = final_knn.predict([x_vec])[0] 
            
        predictions.append(predicted_grade)

    # Print
    print("\nPredictions")
    print("diagram_number,dataset1_grade,dataset2_grade")
    results = {}
    for idx, key in enumerate(sorted_test_keys):
        erd_num, dataset_num = key
        if erd_num not in results:
            results[erd_num] = [None, None]
        results[erd_num][dataset_num-1] = round(predictions[idx], 2)

    for erd_num in sorted(results.keys()):
        ds1 = results[erd_num][0] if results[erd_num][0] is not None else ''
        ds2 = results[erd_num][1] if results[erd_num][1] is not None else ''
        print(f"{erd_num},{ds1},{ds2}")


# Loading Data / Environment Setup

def load_data_and_setup():
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        print("Downloading NLTK punkt data...")
        nltk.download('punkt')
        
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = SCRIPT_DIR 
    
    GRADES_PATH = os.path.join(DATA_DIR, 'ERD_grades.csv')
    DATASET1_PATH = os.path.join(DATA_DIR, 'Dataset1')
    DATASET2_PATH = os.path.join(DATA_DIR, 'Dataset2')

    try:
        if not os.path.exists(GRADES_PATH):
             raise FileNotFoundError
             
        csv = pd.read_csv(GRADES_PATH)
        erd_no = csv['ERD_No'].tolist()
        dataset1_grade = csv['dataset1_grade'].tolist()
        dataset2_grade = csv['dataset2_grade'].tolist()
    except FileNotFoundError:
        print(f"Error: Grades file not found at {GRADES_PATH}. Please ensure ERD_grades.csv is in the same directory as this script.")
        return None, None, None, None, None, None

    train_jsons = {}
    test_jsons = {}
    
    def load_dataset(dataset_path, dataset_num, train_dict, test_dict):
        if not os.path.exists(dataset_path):
             print(f"Warning: Dataset directory not found: {dataset_path}. Skipping.")
             return

        for filename in os.listdir(dataset_path):
            if filename.endswith(".json"):
                try:
                    erd_num = int(filename.split(".")[0])
                    file_path = os.path.join(dataset_path, filename)
                    
                    with open(file_path, "r") as f:
                        json_content = f.read()

                    key = (erd_num, dataset_num)
                    
                    # Check if graded for training / testing split
                    grade_col = f'dataset{dataset_num}_grade'
                    grade_row = csv[csv['ERD_No'] == erd_num]
                    is_graded = not grade_row.empty and pd.notna(grade_row.iloc[0][grade_col])
                    
                    if is_graded:
                        train_dict[key] = json_content
                    else:
                        test_dict[key] = json_content
                except ValueError:
                    pass
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    load_dataset(DATASET1_PATH, 1, train_jsons, test_jsons)
    load_dataset(DATASET2_PATH, 2, train_jsons, test_jsons)

    print(f"Loaded {len(train_jsons)} training documents.")
    print(f"Loaded {len(test_jsons)} test/prediction documents.")

    all_graded_grades = [g for g in dataset1_grade if pd.notna(g)] + \
                        [g for g in dataset2_grade if pd.notna(g)]
    global_mean_grade = np.mean(all_graded_grades) if all_graded_grades else 0.0
    print(f"Calculated Global Mean Grade: {global_mean_grade:.3f}")
    
    return train_jsons, test_jsons, erd_no, dataset1_grade, dataset2_grade, global_mean_grade


if __name__ == '__main__':
    train_jsons, test_jsons, erd_no, dataset1_grade, dataset2_grade, GLOBAL_MEAN_GRADE = load_data_and_setup()
    
    if train_jsons:
        run_approach_2(train_jsons, test_jsons, erd_no, dataset1_grade, dataset2_grade, GLOBAL_MEAN_GRADE)