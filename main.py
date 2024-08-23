import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import syllapy 


try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer data is available.")
except LookupError:
    print("Punkt tokenizer data is not available.")

# Function to load stop words from multiple files
def load_stop_words(directory):
    stop_words = set()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='latin1') as file:
                stop_words.update(file.read().split())
    return stop_words

# Function to load dictionaries from text files
def load_dictionary(filename):
    with open(filename, 'r', encoding='latin1') as file:
        return set(word.strip().lower() for word in file.readlines())

# Load stop words and dictionaries
stop_words_directory = 'StopWords'
stop_words = load_stop_words(stop_words_directory)

positive_words = load_dictionary('MasterDictionary/positive-words.txt')
negative_words = load_dictionary('MasterDictionary/negative-words.txt')

# Function to extract article title and text from a webpage
def extract_article(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            title = soup.find('title').get_text() if soup.find('title') else "No Title"
            article_text = ''
            article_section = soup.find_all('p')
            for paragraph in article_section:
                article_text += paragraph.get_text() + '\n'
            return title, article_text
        else:
            print(f"Failed to fetch the article: {url} (Status code: {response.status_code})")
            return None, None
    except Exception as e:
        print(f"Error fetching article from {url}: {e}")
        return None, None
    

def simple_tokenize(text):
    ## Tokenizes by splitting on whitespace and punctuation
    return re.findall(r'\b\w+\b', text)


# Perform textual analysis
def perform_textual_analysis(text):
    # Tokenization
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
    except (ImportError, LookupError):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        words = simple_tokenize(text)
    
    # Clean tokens
    cleaned_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    # Calculate POSITIVE SCORE and NEGATIVE SCORE
    positive_score = sum(1 for word in cleaned_words if word in positive_words)
    negative_score = sum(1 for word in cleaned_words if word in negative_words)
    
    # Calculate POLARITY SCORE
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    
    # Calculate SUBJECTIVITY SCORE
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)
    
    # Calculate readability metrics
    num_sentences = len(sentences)
    avg_sentence_length = len(cleaned_words) / num_sentences if num_sentences > 0 else 0
    complex_words = [word for word in cleaned_words if syllapy.count(word) > 2]
    percentage_complex_words = len(complex_words) / len(cleaned_words) if len(cleaned_words) > 0 else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(cleaned_words) / num_sentences if num_sentences > 0 else 0
    complex_word_count = len(complex_words)
    word_count = len(cleaned_words)
    syllable_count_per_word = sum(syllapy.count(word) for word in cleaned_words) / word_count if word_count > 0 else 0
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.IGNORECASE))
    avg_word_length = sum(len(word) for word in cleaned_words) / len(cleaned_words) if len(cleaned_words) > 0 else 0
    
    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllable_count_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

# Load the Excel file
df = pd.read_excel('Input.xlsx', engine='openpyxl')
df = df.dropna(subset=['URL_ID', 'URL'])
data_dict = df.set_index('URL_ID')['URL'].to_dict()

# Process each URL and perform textual analysis
results = []
for url_id, link in data_dict.items():
    print(f"Processing URL ID: {url_id}, URL: {link}")
    title, article_text = extract_article(link)
    if title and article_text:
        analysis_results = perform_textual_analysis(article_text)
        results.append({'URL_ID': url_id, 'Title': title, **analysis_results})
    else:
        print(f"Skipping article {url_id} due to extraction failure.")

# Save results to an output Excel file
results_df = pd.DataFrame(results)
results_df.to_excel('Output Data Structure.xlsx', index=False)
print("Textual analysis completed and results saved to 'Copy of Output Data Structure.xlsx'.")
