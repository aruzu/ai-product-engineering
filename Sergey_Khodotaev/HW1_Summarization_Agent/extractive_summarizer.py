import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from heapq import nlargest
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class ExtractiveSummarizer:
    def __init__(self):
        self.stopwords = set(stopwords.words('english') + list(punctuation))
    
    def preprocess(self, text):

        sentences = sent_tokenize(text, language='english')
        

        word_freq = defaultdict(int)
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word not in self.stopwords:
                    word_freq[word] += 1
        
        sentence_scores = defaultdict(float)
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word not in self.stopwords:
                    sentence_scores[sentence] += word_freq[word]
        
        return sentences, sentence_scores
    
    def summarize(self, text, num_sentences=5):
        sentences, sentence_scores = self.preprocess(text)

        lent_sentences = len(sentences)
        print(f"Number of sentences: {lent_sentences}")

        if(lent_sentences < num_sentences):
            num_sentences = max(1, len(sentences) // 2)

        summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        
        summary_sentences.sort(key=lambda x: sentences.index(x))
        
        return ' '.join(summary_sentences)
