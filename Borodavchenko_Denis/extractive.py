from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
import numpy as np

class ExtractiveSummarizer:
    """
    Extractive summarization using TextRank algorithm.
    This is a deterministic approach that selects existing sentences from the text.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def _sentence_similarity(self, sent1, sent2):
        """Calculate cosine similarity between two sentences."""
        words1 = [word.lower() for word in word_tokenize(sent1) if word.isalnum()]
        words2 = [word.lower() for word in word_tokenize(sent2) if word.isalnum()]
        
        # Filter out stop words
        words1 = [word for word in words1 if word not in self.stop_words]
        words2 = [word for word in words2 if word not in self.stop_words]
        
        # Create a set of all words
        all_words = list(set(words1 + words2))
        
        # Create word vectors
        vector1 = [1 if word in words1 else 0 for word in all_words]
        vector2 = [1 if word in words2 else 0 for word in all_words]
        
        # Handle empty vectors
        if sum(vector1) == 0 or sum(vector2) == 0:
            return 0.0
        
        # Calculate cosine similarity
        return 1 - cosine_distance(vector1, vector2)
    
    def _build_similarity_matrix(self, sentences):
        """Build similarity matrix for all sentences."""
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
        
        return similarity_matrix
    
    def summarize(self, text):
        """Generate a summary of the given text."""
        # Split text into sentences
        sentences = sent_tokenize(text)
            
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
            
        # Create a graph and apply PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
            
        # Sort sentences by score and select top ones
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        selected_indices = [ranked_sentences[i][1] for i in range(len(ranked_sentences))]
        selected_indices.sort()  # Preserve original sentence order
        
        # Join selected sentences
        summary = " ".join([sentences[i] for i in selected_indices])
        return summary