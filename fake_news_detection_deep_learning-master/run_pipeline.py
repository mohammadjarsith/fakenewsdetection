import nltk
import sys
import os
import traceback

def setup_nltk():
    print("Setting up NLTK...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download('punkt_tab')

def main():
    setup_nltk()
    print("Importing preprocess module...")
    global preprocess
    import preprocess

    print("Running Preprocessing Pipeline...")
    
    print("1. Selecting Top 8000 Words based on TF-IDF...")
    try:
        preprocess.selectTop8000Words_based_tfidf()
        print("   Done.")
    except Exception as e:
        print(f"   Error in selectTop8000Words_based_tfidf: {e}")
        import traceback
        traceback.print_exc()

    print("2. Sentence Tokenization...")
    try:
        preprocess.sentence_tokenize()
        print("   Done.")
    except Exception as e:
        print(f"   Error in sentence_tokenize: {e}")
        traceback.print_exc()

    print("3. Dividing Dataset into 5 parts...")
    try:
        preprocess.divide_dataset_into_5parts()
        print("   Done.")
    except Exception as e:
        print(f"   Error in divide_dataset_into_5parts: {e}")
        traceback.print_exc()

    print("\nPipeline Execution Complete (Preprocessing Only).")
    print("Note: Deep Learning model training was skipped due to missing TensorFlow and Word2Vec embeddings.")

if __name__ == "__main__":
    main()
