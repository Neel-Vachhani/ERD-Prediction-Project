import os
import json
import re
import math
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
import nltk
from gensim import corpora
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize

def split_camel_case(name):
    # Splits camelCase, pascalcase words into separate words
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

def get_normalized_tokens(json_string, use_stemming=True):
    # Stems and normalizes tokens from JSON string
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
    stemmer = PorterStemmer()

    # Normalize
    for item in raw_terms:
        if not item: continue
        
        item = split_camel_case(item)
        item = item.lower()
        item = re.sub(r'[_\-\/\\]', ' ', item)
        item = re.sub(r'[^a-z\s]', '', item)
        item = re.sub(r'\s+', ' ', item).strip()
        item = item.replace('num', 'number').replace('id', 'identifier')
        item = item.replace('no', 'number') 
        tokens = item.split()
        if use_stemming:
            tokens = [stemmer.stem(token) for token in tokens]

        normalized_tokens.extend(tokens)

    return normalized_tokens


def run_approach_1(train_jsons, test_jsons, erd_no, dataset1_grade, dataset2_grade, GLOBAL_MEAN_GRADE):
    
    print("\nRunning Approach 1")

    new_train_jsons = {key: get_normalized_tokens(json_string, use_stemming=True) 
                       for key, json_string in train_jsons.items()}
    new_test_jsons = {key: get_normalized_tokens(json_string, use_stemming=True) 
                      for key, json_string in test_jsons.items()}
    # Create Dictionary and Bag-of-Words
    corpusDict = corpora.Dictionary({**new_train_jsons, **new_test_jsons}.values())
    train_corpusBow = {key : corpusDict.doc2bow(doc) for key, doc in new_train_jsons.items()}
    test_corpusBow = {key : corpusDict.doc2bow(doc) for key, doc in new_test_jsons.items()}

    total_vocab_size = len(corpusDict)
    total_docs = len(new_train_jsons) + len(new_test_jsons)

    combined_idocument_frequency_matrix = np.zeros((total_vocab_size, total_docs), dtype=np.float32)
    train_term_frequency_matrix = np.zeros((total_vocab_size, len(new_train_jsons)), dtype=np.float32)
    
    sorted_train_keys = sorted(new_train_jsons.keys())
    sorted_test_keys = sorted(new_test_jsons.keys())
    
    for index, key in enumerate(sorted_train_keys):
        for term_id, frequency in train_corpusBow[key]:
            train_term_frequency_matrix[term_id, index] = np.float32(frequency)
            combined_idocument_frequency_matrix[term_id, index] += 1.0 

    test_term_frequency_matrix = np.zeros((total_vocab_size, len(new_test_jsons)), dtype=np.float32)
    for index, key in enumerate(sorted_test_keys):
        combined_index = index + len(new_train_jsons)
        for term_id, frequency in test_corpusBow[key]:
            test_term_frequency_matrix[term_id, index] = np.float32(frequency)
            combined_idocument_frequency_matrix[term_id, combined_index] += 1.0 

    combined_idocument_frequency_matrix = total_docs / combined_idocument_frequency_matrix
    combined_idocument_frequency_matrix[np.isinf(combined_idocument_frequency_matrix)] = 0

    train_idocument_frequency_matrix = combined_idocument_frequency_matrix[:, :len(new_train_jsons)]
    test_idocument_frequency_matrix = combined_idocument_frequency_matrix[:, len(new_train_jsons):]

    # Calc TF-IDF
    lp1 = np.vectorize(lambda x : (0 if (x == 0) else (math.log10(x) + 1)), otypes=[np.float32])
    train_tfidf = lp1(train_term_frequency_matrix) * lp1(train_idocument_frequency_matrix)
    test_tfidf = lp1(test_term_frequency_matrix) * lp1(test_idocument_frequency_matrix)

    # LSA / SVD
    COMBINED_MATRIX = np.concatenate((train_tfidf, test_tfidf), axis=1)
    CONCEPTS = min(75, COMBINED_MATRIX.shape[0], COMBINED_MATRIX.shape[1]) 
    
    if CONCEPTS > 0:
        U, S, VT = svd(COMBINED_MATRIX)
        VT = VT[:CONCEPTS, :] 

        train_concept_document_matrix = VT[:, : train_tfidf.shape[1]]
        test_concept_document_matrix = VT[:, train_tfidf.shape[1] :]
    else:
        print("Warning: Insufficient data for LSA/SVD. Using zero vectors.")
        train_concept_document_matrix = np.zeros((1, len(new_train_jsons)))
        test_concept_document_matrix = np.zeros((1, len(new_test_jsons)))


    X_train = train_concept_document_matrix.T
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
    best_k, best_weights, best_metric = 3, 'distance', 'cosine'

    if X_train.shape[0] >= 5:
        for k in [3, 5, 7, 9, 11]:
            for weights in ['uniform', 'distance']:
                for metric in ['euclidean', 'cosine']:
                    knn = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric)
                    cv_rmse = np.sqrt(-cross_val_score(knn, X_train, y_train, cv=min(5, len(y_train)),
                                                       scoring='neg_mean_squared_error').mean())
                    if cv_rmse < best_rmse:
                        best_rmse = cv_rmse
                        best_k, best_weights, best_metric = k, weights, metric

    print(f"\nApproach 1 Tuning: K={best_k}, weights={best_weights}, metric={best_metric}, CV RMSE={best_rmse:.3f}")

    # Final training
    final_knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights, metric=best_metric)
    final_knn.fit(X_train, y_train)

    X_test = test_concept_document_matrix.T
    predictions = []
    q_threshold = 0.5

    for i, x_vec in enumerate(X_test):
        if X_train.shape[0] < best_k: # Too little data
            predicted_grade = GLOBAL_MEAN_GRADE
        else:
            k_predict = max(best_k, 2)
            distances, indices = final_knn.kneighbors([x_vec], n_neighbors=k_predict)
            similarities = 1 - distances[0][:best_k]
            valid_indices = indices[0][:best_k][similarities >= q_threshold]

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
    print("\nApproach 1 Final Predictions")
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


# Data loading / environment setup

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
        """Helper to load files from a single dataset directory."""
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
        run_approach_1(train_jsons, test_jsons, erd_no, dataset1_grade, dataset2_grade, GLOBAL_MEAN_GRADE)