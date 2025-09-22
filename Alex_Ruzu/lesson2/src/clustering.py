"""
User Clustering and Persona Extraction
Advanced k-means clustering with TF-IDF vectorization and persona generation from user segments
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import time
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Configuration from environment variables
MAX_CLUSTER_PERSONAS = int(os.getenv('MAX_CLUSTER_PERSONAS', '5'))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('stopwords')


class UserClusteringAnalyzer:
    """Basic clustering-based feature extraction system"""

    def __init__(self, csv_path="data/viber.csv", output_dir="docs/clustering_results"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

        # Viber-specific stop words
        self.viber_stopwords = {
            'viber', 'app', 'good', 'great', 'bad', 'nice', 'ok', 'okay',
            'use', 'using', 'used', 'like', 'love', 'hate', 'work', 'working',
            'works', 'get', 'got', 'one', 'time', 'much', 'many', 'way',
            'thing', 'things', 'really', 'very', 'still', 'also', 'well',
            'make', 'makes', 'made', 'need', 'needs', 'want', 'wants'
        }

        os.makedirs(output_dir, exist_ok=True)

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^\w\s\.\!\?\,\-]', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        return text

    def load_and_preprocess_data(self, limit: int = None) -> pd.DataFrame:
        """Load and preprocess the Viber reviews dataset"""
        logger.debug(f"Loading data from {self.csv_path}...")

        df = pd.read_csv(self.csv_path)
        logger.debug(f"Loaded {len(df)} reviews")

        if limit:
            df = df.head(limit)
            logger.debug(f"Using first {len(df)} reviews")

        # Clean data
        df = df.dropna(subset=['content'])
        df['content_clean'] = df['content'].apply(self.preprocess_text)
        df = df[df['content_clean'].str.len() > 10]

        # Add sentiment analysis
        sentiments = []
        for text in df['content_clean']:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            sentiments.append({
                'compound': sentiment['compound'],
                'positive': sentiment['pos'],
                'negative': sentiment['neg'],
                'neutral': sentiment['neu']
            })

        df['sentiment'] = sentiments
        df['sentiment_label'] = df['sentiment'].apply(
            lambda x: 'positive' if x['compound'] > 0.1
            else 'negative' if x['compound'] < -0.1
            else 'neutral'
        )

        logger.debug(f"Final dataset: {len(df)} reviews after preprocessing")
        return df

    def create_tfidf_features(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Create TF-IDF features for clustering"""
        logger.debug("Creating TF-IDF features...")

        all_stopwords = self.stop_words.union(self.viber_stopwords)

        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(all_stopwords),
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=3
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.debug(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        return tfidf_matrix.toarray(), vectorizer

    def find_optimal_clusters(self, features: np.ndarray, min_k=3, max_k=15) -> int:
        """Find optimal number of clusters using silhouette score"""
        logger.debug("Finding optimal number of clusters...")

        silhouette_scores = []
        k_range = range(min_k, min(max_k + 1, len(features) // 10))

        best_k = min_k
        best_score = -1

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)

                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(score)
                    logger.debug(f"K={k}: Silhouette Score = {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_k = k
                else:
                    silhouette_scores.append(-1)

            except Exception as e:
                logger.warning(f"Error for k={k}: {e}")
                silhouette_scores.append(-1)

        logger.debug(f"Optimal K: {best_k} (Silhouette Score: {best_score:.4f})")
        return best_k

    def perform_clustering(self, features: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(features)

        logger.debug(f"Performing K-means clustering with {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        logger.debug("Cluster distribution:")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.debug(f"  Cluster {cluster_id}: {count} reviews")

        return cluster_labels

    def extract_cluster_keywords(self, cluster_texts: List[str], vectorizer: TfidfVectorizer, top_k=10) -> List[str]:
        """Extract top keywords for a cluster using TF-IDF"""
        if len(cluster_texts) < 3:
            return []

        try:
            # Create TF-IDF for cluster texts
            cluster_tfidf = vectorizer.transform(cluster_texts)
            mean_scores = np.mean(cluster_tfidf.toarray(), axis=0)
            feature_names = vectorizer.get_feature_names_out()

            # Get top keywords
            top_indices = np.argsort(mean_scores)[::-1]
            keywords = []

            all_stopwords = self.stop_words.union(self.viber_stopwords)

            for idx in top_indices[:top_k * 2]:
                keyword = feature_names[idx]
                if (len(keyword) >= 3 and
                    keyword not in all_stopwords and
                    not keyword.isdigit() and
                    sum(c.isalpha() for c in keyword) / len(keyword) >= 0.6):
                    keywords.append(keyword)

            return keywords[:top_k]

        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            return []

    def extract_feature_requests(self, cluster_data: pd.DataFrame) -> List[Dict]:
        """Extract potential feature requests from cluster reviews"""
        feature_patterns = {
            'ui_improvements': [
                r'interface', r'ui\b', r'design', r'layout', r'button', r'menu',
                r'navigation', r'easier', r'simpler', r'confusing', r'difficult'
            ],
            'functionality_requests': [
                r'feature', r'function', r'add', r'include', r'implement',
                r'wish', r'would like', r'should have', r'missing', r'need'
            ],
            'performance_issues': [
                r'slow', r'fast', r'speed', r'lag', r'freeze', r'crash',
                r'performance', r'optimize', r'improve', r'better'
            ],
            'bug_reports': [
                r'bug', r'error', r'problem', r'issue', r'fix', r'broken',
                r'not working', r'doesn\'t work', r'fails', r'glitch'
            ],
            'sync_backup': [
                r'sync', r'backup', r'restore', r'transfer', r'switch device',
                r'cross device', r'synchronize', r'data loss'
            ],
            'ads_monetization': [
                r'ads?', r'advertisement', r'premium', r'subscription', r'free',
                r'paid', r'spam', r'annoying', r'remove ads'
            ],
            'activation_login': [
                r'activation', r'activate', r'login', r'log in', r'sign up',
                r'register', r'account', r'pin', r'verification'
            ]
        }

        feature_requests = []

        for category, patterns in feature_patterns.items():
            matching_reviews = []

            for _, row in cluster_data.iterrows():
                text = row['content_clean']
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                    matching_reviews.append({
                        'reviewId': row['reviewId'],
                        'content': row['content'],
                        'score': row['score'],
                        'sentiment': row['sentiment']['compound']
                    })

            if matching_reviews:
                avg_sentiment = np.mean([r['sentiment'] for r in matching_reviews])
                frequency = len(matching_reviews) / len(cluster_data)

                feature_requests.append({
                    'category': category,
                    'frequency': frequency,
                    'count': len(matching_reviews),
                    'average_sentiment': avg_sentiment,
                    'priority_score': frequency * (1 - avg_sentiment),
                    'sample_reviews': matching_reviews[:3]
                })

        feature_requests.sort(key=lambda x: x['priority_score'], reverse=True)
        return feature_requests

    def analyze_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray, vectorizer: TfidfVectorizer) -> Dict[int, Dict]:
        """Analyze clusters and extract features"""
        logger.debug("Analyzing clusters...")

        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels

        cluster_analysis = {}

        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]

            if len(cluster_data) < 10:
                continue

            logger.debug(f"Analyzing Cluster {cluster_id} ({len(cluster_data)} reviews)...")

            # Extract keywords
            keywords = self.extract_cluster_keywords(
                cluster_data['content_clean'].tolist(),
                vectorizer
            )

            # Sentiment analysis
            sentiment_dist = cluster_data['sentiment_label'].value_counts().to_dict()
            avg_sentiment = cluster_data['sentiment'].apply(lambda x: x['compound']).mean()

            # Rating analysis
            rating_dist = cluster_data['score'].value_counts().sort_index().to_dict()
            avg_rating = cluster_data['score'].mean()

            # Sample reviews
            sample_reviews = []
            for sentiment in ['negative', 'neutral', 'positive']:
                sentiment_data = cluster_data[cluster_data['sentiment_label'] == sentiment]
                if len(sentiment_data) > 0:
                    sample = sentiment_data.sample(n=min(2, len(sentiment_data)), random_state=42)
                    for _, row in sample.iterrows():
                        sample_reviews.append({
                            'reviewId': row['reviewId'],
                            'content': row['content'],
                            'score': row['score'],
                            'sentiment': sentiment
                        })

            # Extract feature requests
            feature_requests = self.extract_feature_requests(cluster_data)

            # Calculate urgency score
            size_factor = min(len(cluster_data) / 100, 1.0)
            sentiment_factor = max(0, -avg_sentiment)
            rating_factor = max(0, (3 - avg_rating) / 2)
            urgency_score = size_factor * 0.4 + sentiment_factor * 0.4 + rating_factor * 0.2

            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'keywords': keywords,
                'sentiment_distribution': sentiment_dist,
                'average_sentiment': avg_sentiment,
                'rating_distribution': rating_dist,
                'average_rating': avg_rating,
                'sample_reviews': sample_reviews[:5],
                'feature_requests': feature_requests,
                'urgency_score': min(urgency_score, 1.0),
                'raw_data': cluster_data  # Store for persona generation
            }

        return cluster_analysis

    def generate_personas_from_clusters(self, cluster_analysis: Dict[int, Dict]) -> List[Dict]:
        """Generate personas from cluster analysis results"""
        logger.debug("Generating personas from clusters...")

        personas = []

        # Sort clusters by size to prioritize major user segments
        sorted_clusters = sorted(cluster_analysis.items(),
                               key=lambda x: x[1]['size'], reverse=True)

        # Define persona templates based on user behavior patterns
        persona_templates = {
            'frustrated_user': {
                'age_range': (25, 45),
                'roles': ['Marketing Manager', 'Sales Executive', 'Business Professional', 'Project Manager'],
                'locations': ['New York, USA', 'London, UK', 'Toronto, Canada', 'Sydney, Australia']
            },
            'casual_user': {
                'age_range': (18, 35),
                'roles': ['Student', 'Designer', 'Teacher', 'Freelancer'],
                'locations': ['San Francisco, USA', 'Berlin, Germany', 'Tokyo, Japan', 'Vancouver, Canada']
            },
            'business_user': {
                'age_range': (30, 50),
                'roles': ['CEO', 'Team Lead', 'Consultant', 'Director'],
                'locations': ['Chicago, USA', 'Singapore', 'Amsterdam, Netherlands', 'Melbourne, Australia']
            }
        }

        for i, (cluster_id, analysis) in enumerate(sorted_clusters):
            if analysis['size'] < 20:  # Skip very small clusters
                continue


            # Determine persona type based on cluster characteristics
            persona_type = self._determine_persona_type(analysis)
            template = persona_templates.get(persona_type, persona_templates['casual_user'])

            # Generate persona name and demographics
            persona_name = self._generate_persona_name(cluster_id, analysis['keywords'][:2])
            age = np.random.randint(template['age_range'][0], template['age_range'][1])
            age = int(age)
            role = np.random.choice(template['roles'])
            role = str(role)
            location = np.random.choice(template['locations'])
            location = str(location)

            # Create persona background based on cluster characteristics
            background = self._create_persona_background(analysis, role, persona_type)

            # Extract pain points from feature requests
            pain_points = []
            for feature in analysis['feature_requests'][:5]:  # Top 5 feature requests
                pain_points.append(self._feature_to_pain_point(feature))

            # Create needs from pain points
            needs = []
            for feature in analysis['feature_requests'][:3]:  # Top 3 needs
                needs.append(self._feature_to_need(feature))

            # Generate usage pattern description
            usage_pattern = self._create_usage_pattern(analysis)

            # Get evidence review IDs
            evidence_ids = [review['reviewId'] for review in analysis['sample_reviews'][:3]]

            persona = {
                'name': persona_name,
                'profile': {
                    'age': age,
                    'role': role,
                    'location': location
                },
                'background': background,
                'pain_points': pain_points,
                'needs': needs,
                'usage_pattern': usage_pattern,
                'cluster_info': {
                    'cluster_id': cluster_id,
                    'cluster_size': analysis['size'],
                    'urgency_score': analysis['urgency_score'],
                    'average_rating': analysis['average_rating'],
                    'average_sentiment': analysis['average_sentiment']
                },
                'evidence': evidence_ids
            }

            personas.append(persona)

            # Limit to 5 personas max
            if len(personas) >= MAX_CLUSTER_PERSONAS:
                break

        logger.debug(f"Generated {len(personas)} personas from {len(sorted_clusters)} clusters")
        return personas

    def _determine_persona_type(self, analysis: Dict) -> str:
        """Determine persona type based on cluster characteristics"""
        avg_rating = analysis['average_rating']
        avg_sentiment = analysis['average_sentiment']

        # Look for business-related keywords
        business_keywords = ['work', 'business', 'team', 'client', 'professional', 'office']
        has_business_context = any(keyword in ' '.join(analysis['keywords']).lower()
                                 for keyword in business_keywords)

        # Look for activation/technical issues
        tech_keywords = ['activation', 'failed', 'error', 'bug', 'problem']
        has_tech_issues = any(keyword in ' '.join(analysis['keywords']).lower()
                            for keyword in tech_keywords)

        if has_business_context or (avg_rating <= 3.0 and has_tech_issues):
            if avg_sentiment < -0.1:
                return 'frustrated_user'
            else:
                return 'business_user'
        else:
            return 'casual_user'

    def _generate_persona_name(self, cluster_id: int, keywords: List[str]) -> str:
        """Generate a unique persona name based on cluster characteristics"""
        # Generate unique names based on primary keywords and cluster ID for uniqueness
        base_names = []

        if any(word in ['ads', 'account', 'calls'] for word in keywords):
            base_names = ['Ad-Weary Alice', 'Account Advocate Amy', 'Call-Cautious Carl']
        elif any(word in ['call', 'activate', 'phone'] for word in keywords):
            base_names = ['Activation Andy', 'Phone Setup Paul', 'Call-Ready Chris']
        elif any(word in ['messages', 'spam', 'notifications'] for word in keywords):
            base_names = ['Message Manager Mark', 'Notification Nancy', 'Spam-Savvy Sam']
        elif any(word in ['number', 'phone number', 'new phone'] for word in keywords):
            base_names = ['Phone Setup Pete', 'Number Nina', 'Device Danny']
        elif any(word in ['apps', 'code', 'activation'] for word in keywords):
            base_names = ['Technical Tom', 'Code-Curious Cathy', 'App Annie']
        elif any(word in ['update', 'last update', 'latest'] for word in keywords):
            base_names = ['Update-Unhappy Uma', 'Version Victor', 'Upgrade Ursula']
        elif any(word in ['help', 'please', 'support'] for word in keywords):
            base_names = ['Help-Seeking Helen', 'Support Steve', 'Assistance Alan']
        elif any(word in ['best', 'ever', 'excellent'] for word in keywords):
            base_names = ['Happy Hannah', 'Satisfied Sarah', 'Delighted Dave']
        elif any(word in ['useful', 'application', 'thank', 'good', 'feature'] for word in keywords):
            base_names = ['Feature-Focused Frank', 'Utility Una', 'Grateful Grace']

        if base_names:
            # Use cluster_id to pick a consistent but unique name from the list
            return base_names[cluster_id % len(base_names)]
        else:
            return f'User Segment {cluster_id}'

    def _create_persona_background(self, analysis: Dict, role: str, persona_type: str) -> str:
        """Create persona background based on cluster analysis"""
        primary_issues = analysis['keywords'][:3]

        if persona_type == 'frustrated_user':
            return f"A {role.lower()} who relies on Viber for important communication but faces frequent issues with {', '.join(primary_issues)}. Values reliability and efficiency in communication tools."
        elif persona_type == 'business_user':
            return f"A {role.lower()} who uses Viber for professional communication with clients and team members. Needs dependable features and is particularly concerned about {', '.join(primary_issues)}."
        else:
            return f"A {role.lower()} who uses Viber primarily for personal communication with friends and family. Generally enjoys the app but has encountered issues with {', '.join(primary_issues)}."

    def _feature_to_pain_point(self, feature: Dict) -> str:
        """Convert feature request to user pain point"""
        category = feature['category']

        pain_point_mapping = {
            'activation_login': 'Cannot reliably activate or log into accounts',
            'ads_monetization': 'Too many intrusive advertisements interrupting usage',
            'bug_reports': 'App crashes, freezes, or behaves unexpectedly',
            'performance_issues': 'Slow app performance and poor responsiveness',
            'ui_improvements': 'Confusing or difficult-to-use interface elements',
            'sync_backup': 'Messages and data not syncing properly across devices',
            'functionality_requests': 'Missing important features for daily use'
        }

        return pain_point_mapping.get(category, f"Issues with {category.replace('_', ' ')}")

    def _feature_to_need(self, feature: Dict) -> str:
        """Convert feature request to user need"""
        category = feature['category']

        need_mapping = {
            'activation_login': 'Reliable and simple account activation process',
            'ads_monetization': 'Less intrusive advertising or affordable premium options',
            'bug_reports': 'Stable and reliable app performance',
            'performance_issues': 'Fast and responsive user experience',
            'ui_improvements': 'Intuitive and user-friendly interface',
            'sync_backup': 'Seamless data synchronization across all devices',
            'functionality_requests': 'Enhanced features that improve daily communication'
        }

        return need_mapping.get(category, f"Better {category.replace('_', ' ')}")

    def _create_usage_pattern(self, analysis: Dict) -> str:
        """Create usage pattern description based on cluster analysis"""
        size_percent = analysis['size']  # Will be calculated as percentage later
        avg_rating = analysis['average_rating']
        sentiment = analysis['average_sentiment']
        top_issues = analysis['keywords'][:3]

        # Determine usage frequency
        if analysis['size'] > 100:
            frequency = "heavy user"
        elif analysis['size'] > 50:
            frequency = "regular user"
        else:
            frequency = "occasional user"

        # Determine satisfaction level
        if avg_rating <= 2.5:
            satisfaction = "highly dissatisfied"
        elif avg_rating <= 3.5:
            satisfaction = "somewhat frustrated"
        else:
            satisfaction = "generally satisfied"

        return f"{frequency.title()} who is {satisfaction} with the app, primarily due to issues with {', '.join(top_issues)}."

    def save_results(self, cluster_analysis: Dict[int, Dict], df: pd.DataFrame):
        """Save comprehensive results including cluster-based personas"""
        logger.debug("Saving results...")

        # Generate personas from clusters
        personas = self.generate_personas_from_clusters(cluster_analysis)

        # Convert numpy int keys to regular ints for JSON serialization
        clusters_serializable = {}
        for k, v in cluster_analysis.items():
            # Remove raw_data before serialization (too large)
            v_clean = {key: value for key, value in v.items() if key != 'raw_data'}
            clusters_serializable[str(k)] = v_clean

        # Detailed JSON results
        results_data = {
            'metadata': {
                'total_reviews': len(df),
                'total_clusters': len(cluster_analysis),
                'total_personas': len(personas),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'clusters': clusters_serializable,
            'personas': personas
        }

        with open(f'{self.output_dir}/clustering_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Save personas separately
        with open(f'{self.output_dir}/cluster_personas.json', 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Feature requests summary
        all_features = []
        for cluster_id, analysis in cluster_analysis.items():
            for feature in analysis['feature_requests']:
                all_features.append({
                    'cluster_id': cluster_id,
                    'cluster_size': analysis['size'],
                    'urgency_score': analysis['urgency_score'],
                    **feature
                })

        all_features.sort(key=lambda x: x['priority_score'], reverse=True)

        with open(f'{self.output_dir}/feature_requests_prioritized.json', 'w', encoding='utf-8') as f:
            json.dump(all_features, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Human-readable report with personas
        self._save_report(cluster_analysis, df, personas)

        logger.debug(f"Results saved to {self.output_dir}/")
        logger.debug(f"Generated {len(personas)} cluster-based personas")

    def _save_report(self, cluster_analysis: Dict[int, Dict], df: pd.DataFrame, personas: List[Dict] = None):
        """Save human-readable markdown report"""
        report = []
        report.append("# Viber User Feedback Clustering Analysis")
        report.append(f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Reviews Analyzed:** {len(df)}")
        report.append(f"**Clusters Identified:** {len(cluster_analysis)}")
        if personas:
            report.append(f"**Personas Generated:** {len(personas)}")

        # Sort clusters by urgency
        sorted_clusters = sorted(cluster_analysis.items(),
                               key=lambda x: x[1]['urgency_score'], reverse=True)

        report.append("\n## Executive Summary")
        report.append("\n### Most Urgent Issues:")
        for i, (cluster_id, analysis) in enumerate(sorted_clusters[:5]):
            keywords_str = ", ".join(analysis['keywords'][:5])
            report.append(f"{i+1}. **Cluster {cluster_id}** ({analysis['size']} reviews): {keywords_str}")
            report.append(f"   - Urgency Score: {analysis['urgency_score']:.3f}")
            report.append(f"   - Avg Rating: {analysis['average_rating']:.1f}/5")

        # Top feature requests across all clusters
        all_features = []
        for cluster_id, analysis in cluster_analysis.items():
            for feature in analysis['feature_requests']:
                all_features.append({
                    'cluster_id': cluster_id,
                    'cluster_size': analysis['size'],
                    **feature
                })

        all_features.sort(key=lambda x: x['priority_score'], reverse=True)

        report.append("\n### Top Feature Requests:")
        for i, feature in enumerate(all_features[:10]):
            report.append(f"{i+1}. **{feature['category'].replace('_', ' ').title()}** (Cluster {feature['cluster_id']})")
            report.append(f"   - {feature['count']} mentions, Priority Score: {feature['priority_score']:.3f}")

        # Personas section
        if personas:
            report.append("\n## Cluster-Based User Personas")
            report.append("\nGenerated from review clustering analysis, representing actual user segments:")

            for i, persona in enumerate(personas):
                report.append(f"\n### {i+1}. {persona['name']}")
                report.append(f"**Profile:** {persona['profile']['age']} years old, {persona['profile']['role']} from {persona['profile']['location']}")
                report.append(f"**Background:** {persona['background']}")
                report.append(f"**Usage Pattern:** {persona['usage_pattern']}")

                report.append("**Primary Pain Points:**")
                for pain in persona['pain_points']:
                    report.append(f"- {pain}")

                report.append("**Key Needs:**")
                for need in persona['needs']:
                    report.append(f"- {need}")

                cluster_info = persona['cluster_info']
                report.append(f"**Data Source:** Cluster {cluster_info['cluster_id']} ({cluster_info['cluster_size']} reviews, {cluster_info['average_rating']:.1f}/5 rating)")
                report.append(f"**Evidence:** Based on reviews {', '.join(map(str, persona['evidence']))}")

        # Detailed cluster analysis
        report.append("\n## Detailed Cluster Analysis")

        for cluster_id, analysis in sorted_clusters:
            report.append(f"\n### Cluster {cluster_id}: {', '.join(analysis['keywords'][:3]).title()}")
            report.append(f"**Size:** {analysis['size']} reviews ({analysis['size']/len(df)*100:.1f}%)")
            report.append(f"**Urgency Score:** {analysis['urgency_score']:.3f}")
            report.append(f"**Keywords:** {', '.join(analysis['keywords'])}")
            report.append(f"**Average Rating:** {analysis['average_rating']:.1f}/5")
            report.append(f"**Average Sentiment:** {analysis['average_sentiment']:.3f}")

            # Top feature requests
            if analysis['feature_requests']:
                report.append(f"**Top Feature Requests:**")
                for i, feature in enumerate(analysis['feature_requests'][:3]):
                    report.append(f"  {i+1}. **{feature['category'].replace('_', ' ').title()}** - {feature['count']} mentions")

            # Sample reviews
            report.append(f"**Sample Reviews:**")
            for i, sample in enumerate(analysis['sample_reviews'][:3]):
                sentiment_emoji = "😊" if sample['sentiment'] == "positive" else "😢" if sample['sentiment'] == "negative" else "😐"
                content_preview = sample['content'][:100] + ('...' if len(sample['content']) > 100 else '')
                report.append(f"  {i+1}. {sentiment_emoji} *\"{content_preview}\"*")
                report.append(f"     (Score: {sample['score']}/5)")

            report.append("\n---")

        with open(f'{self.output_dir}/clustering_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

    def run_analysis(self, limit: int = None) -> Dict[int, Dict]:
        """Run the complete clustering analysis"""
        logger.debug("Starting clustering analysis")
        start_time = time.time()

        # Load and preprocess data
        df = self.load_and_preprocess_data(limit=limit)

        # Create TF-IDF features
        tfidf_features, vectorizer = self.create_tfidf_features(df['content_clean'].tolist())

        # Perform clustering
        cluster_labels = self.perform_clustering(tfidf_features)

        # Analyze clusters
        cluster_analysis = self.analyze_clusters(df, cluster_labels, vectorizer)

        # Save results
        self.save_results(cluster_analysis, df)

        execution_time = time.time() - start_time
        logger.debug(f"Clustering analysis completed in {execution_time:.2f} seconds")

        return cluster_analysis


def main():
    """Main execution function"""
    extractor = UserClusteringAnalyzer()

    # Test on first 200 reviews
    logger.info("Phase 1: Testing on first 200 reviews...")
    test_results = extractor.run_analysis(limit=200)

    if len(test_results) >= 3:
        logger.info(f"Test successful! Found {len(test_results)} meaningful clusters.")
        logger.info("Top clusters by urgency:")
        sorted_clusters = sorted(test_results.items(), key=lambda x: x[1]['urgency_score'], reverse=True)
        for i, (cluster_id, analysis) in enumerate(sorted_clusters[:3]):
            keywords = ', '.join(analysis['keywords'][:5])
            logger.info(f"{i+1}. Cluster {cluster_id}: {keywords} ({analysis['size']} reviews)")

        proceed = input("\nProceed with full dataset (10,000 reviews)? (y/n): ")
        if proceed.lower() in ['y', 'yes']:
            logger.info("Phase 2: Running on full dataset...")
            full_results = extractor.run_analysis(limit=None)
            logger.info(f"Full analysis complete! Found {len(full_results)} clusters total.")
        else:
            logger.info("Stopping at test results.")
    else:
        logger.info(f"Test results show only {len(test_results)} clusters. Consider adjusting parameters.")


if __name__ == "__main__":
    main()