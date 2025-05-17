from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import os
import hashlib
import pickle
from collections import Counter, defaultdict
from functools import lru_cache


WORD_PATTERN = re.compile(r'\b[\w\'-_]+\b|\S')

@lru_cache(maxsize=None)
def _preprocess_text(text):
    return ' '.join(WORD_PATTERN.findall(text.lower()))

class KeywordFilter:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

    def initialize_vectorizer(self, corpus):
        corpus_hash = hashlib.md5(str(corpus).encode()).hexdigest()
        cache_file = f'vectorizer_cache_{corpus_hash}.pkl'

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.vectorizer = cached_data['vectorizer']
                self.tfidf_matrix = cached_data['tfidf_matrix']
                self.feature_names = cached_data['feature_names']
        else:
            max_ngram = 4
            corpus = [preprocess_text(text) for text in corpus]
            self.vectorizer = TfidfVectorizer(ngram_range=(1, max_ngram),stop_words='english')

            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.feature_names = self.vectorizer.get_feature_names_out()

            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'tfidf_matrix': self.tfidf_matrix,
                    'feature_names': self.feature_names
                }, f)

    def rank_by_importance(self,keywords,context:str,funcname):
        if self.tfidf_matrix is None:
            raise ValueError("Vectorizer not initialized. Call initialize_vectorizer first.")

        funcname_index = context.rfind('>>>')
        if funcname_index != -1:
            funcname_base = context[funcname_index+3:].split('(')[0].strip()
        else:
            function_name_part = context.split('=')[0]
            function_name_part = function_name_part.split('assert')[-1]
            if function_name_part.startswith(' math.'):
                funcname_base= function_name_part.split('(')[1]
            elif function_name_part.startswith(' set(') :
                funcname_base = function_name_part.split('(')[1]
            elif ' not ' in function_name_part:
                funcname_base = function_name_part.replace(' not ','')
                funcname_base = funcname_base.split('(')[0]
            else:
                funcname_base = function_name_part.split('(')[0]
            
        context = re.sub(r'\s+',' ',context)
        processed_context = context.lower()
        local_tf = self._calculate_local_tf(processed_context)

        global_idf = dict(zip(self.feature_names, self.vectorizer.idf_))

        in_vocabulary = []
        only_in_context = []
        function_keywords = []
        abstrct_keywords = []
        process_keywords = [key.lower() for key in keywords]
        for keyword in process_keywords:
            if keyword in self.feature_names and keyword in context.lower():
                tf = local_tf.get(keyword, 0)
                idf = global_idf[keyword]
                tfidf = tf * idf
                in_vocabulary.append((keyword, tfidf))
            elif keyword not in self.feature_names and  keyword in context.lower() and (keyword != funcname_base.lower() and keyword != funcname.lower()):
                tf = local_tf.get(keyword, 0)
                only_in_context.append((keyword, tf))
            elif keyword not in self.feature_names and keyword in context.lower() and (keyword == funcname_base.lower() or keyword == funcname.lower()):
                function_keywords.append((keyword, 2))  
            elif keyword not in self.feature_names and keyword not in context.lower():
                abstrct_keywords.append((keyword, 3)) 
            else:  
                abstrct_keywords.append((keyword, 4))

        only_in_context.sort(key=lambda x: x[1], reverse=True)
        in_vocabulary.sort(key=lambda x: x[1], reverse=True)
        
        sorted_keywords = [item[0] for item in abstrct_keywords]+ [item[0] for item in only_in_context] + [item[0] for item in in_vocabulary] + [item[0] for item in function_keywords]  
    
        return sorted_keywords

    
    
    def _calculate_local_tf(self, text):
        
        words = WORD_PATTERN.findall(text)
        max_n = 4
        ngram_counts = defaultdict(Counter)
        
        for n in range(1, max_n + 1):
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts[n].update(ngrams)
        
        total_counts = {n: sum(counts.values()) for n, counts in ngram_counts.items()}
        
        tf = {}
        for n, counts in ngram_counts.items():
            for ngram, count in counts.items():
                tf[ngram] = count / total_counts[n]
        
        return tf
        
    def get_top_words(self, n=None, output_file='tf-idf.txt'):
        if self.tfidf_matrix is None:
            raise ValueError("Vectorizer not initialized. Call initialize_vectorizer first.")

        avg_tfidf = np.array(self.tfidf_matrix.mean(axis=0)).flatten()

        word_tfidf_pairs = sorted(zip(self.feature_names, avg_tfidf), key=lambda x: x[1], reverse=True)
        if n is not None:
            word_tfidf_pairs = word_tfidf_pairs[:n]
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for word, tfidf in word_tfidf_pairs:
                    f.write(f'{word}: {tfidf}\n')

        return word_tfidf_pairs



def preprocess_text(text):
    text =  re.sub('[^a-zA-Z]', ' ', text)
    return text

def process_keywords(text):
    flags = ['[Keyword]:','[Keyword_3]:','[Keyword_2]:','[Keyword_1]:','[Formalized explanation]:']
    for flag in flags:
        text = text.replace(flag,'')
    return text

def extract_keywords(output):
    keyword_list = []
    outputs = []
    for line in output.split('\n'):
        if len(line.strip())!=0:
            if 'Here are the top 3 keywords with their explanations' in line:
                continue
            elif 'Here is the analysis of the' in line:
                continue
            elif line.strip().startswith('Here '):
                continue
            outputs.append(f'{process_keywords(line).strip()}')
    output = '\n'.join(outputs)

    for line in output.split('\n'):
        if ':' in line:
            keyword = line.split(':')[0]
            keyword = keyword.strip().strip('[').strip(']')
            keyword = keyword.strip().strip('*')
            keyword = keyword.strip().strip('/')
            keyword_list.append(keyword.lower())
    return keyword_list


def format_output(output,filter_keywords):
    def get_index(line):
        if ':' in line:
            keyword_part = line.split(':')[0]
            keyword_part = keyword_part.strip().strip('[').strip(']')
            keyword_part = keyword_part.strip().strip('*')
            keyword_part = keyword_part.strip().strip('/')
            for keyword in filter_keywords:
                if keyword == keyword_part.lower():
                    return filter_keywords.index(keyword)
        return None

    res_dic = {}
    items = output.split('\n')
    process_items = []
    for i,line in enumerate(items):
        if line.strip().startswith('-'):
            if process_items:
                top_item = process_items[-1]
                top_item = top_item + '\n' + line
                process_items[-1] = top_item
            else:
                process_items.append(line)
        else:
            process_items.append(line)

    for i, line in enumerate(process_items):
        if len(line.strip())!=0:
            line_index = get_index(line)
            if line_index is not None:
                res_dic[line] = line_index
                

    res = [k for k, v in sorted(res_dic.items(), key=lambda item: item[1])]

    return '\n'.join(res)


