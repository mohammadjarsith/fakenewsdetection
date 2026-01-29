import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def load_data(data_dir):
    """
    Loads data from the train_test_data structure created by preprocess.py
    Structure: data_dir/train/file.txt and data_dir/test/file.txt
    """
    print(f"Loading data from {data_dir}...")
    
    # Load Ground Truth to get labels
    ground_truth_file = 'snopes_ground_truth.csv'
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Ground truth file {ground_truth_file} not found.")
    
    df_gt = pd.read_csv(ground_truth_file)
    # Map URLs to labels (True/False)
    # The csv has columns: snopes_page, claim_label, ...
    # claim_label is boolean or string 'true'/'false'
    
    url_to_label = {}
    for index, row in df_gt.iterrows():
        label = str(row['claim_label']).lower()
        if label == 'true':
            url_to_label[row['snopes_page']] = 1 # Real News
        else:
            url_to_label[row['snopes_page']] = 0 # Fake News
            
    def read_folder(folder_path):
        texts = []
        labels = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        print(f"  Found {len(files)} files in {folder_path}")
        
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            if not lines:
                continue
                
            # First line is URL
            url = lines[0].strip()
            
            # Remaining lines are content
            content = " ".join([l.strip() for l in lines[1:]])
            
            if url in url_to_label:
                texts.append(content)
                labels.append(url_to_label[url])
            else:
                # Fallback: try to guess label from filename if available or skip
                # preprocess.py logic: creates file based on listdir order. 
                # Ideally URL should match. 
                pass
                
        return texts, labels

    train_texts, train_labels = read_folder(os.path.join(data_dir, 'train'))
    test_texts, test_labels = read_folder(os.path.join(data_dir, 'test'))
    
    return train_texts, train_labels, test_texts, test_labels

def main():
    # Use the first fold (data_0)
    data_dir = 'train_test_data/data_0'
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist. Did you run 'run_pipeline.py'?")
        return

    try:
        X_train_text, y_train, X_test_text, y_test = load_data(data_dir)
        
        print(f"Train size: {len(X_train_text)}")
        print(f"Test size: {len(X_test_text)}")
        
        if len(X_train_text) == 0:
            print("No training data found. Check preprocess.py output.")
            return

        # Vectorize
        print("Vectorizing text (TF-IDF)...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        # Train Model
        print("Training Logistic Regression Model...")
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict
        print("Predicting...")
        y_pred = clf.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        print("\n" + "="*30)
        print(f"RESULTS (Lite Version)")
        print("="*30)
        print(f"Accuracy: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Inference Demo
        print("\n" + "="*30)
        print("INFERENCE DEMO")
        print("="*30)
        print("Sample predictions from test set:")
        for i in range(min(5, len(X_test_text))):
            text_snippet = X_test_text[i][:100] + "..."
            truth = "Real" if y_test[i] == 1 else "Fake"
            pred = "Real" if y_pred[i] == 1 else "Fake"
            print(f"Text: {text_snippet}")
            print(f"  Truth: {truth} | Predicted: {pred}")
            print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
