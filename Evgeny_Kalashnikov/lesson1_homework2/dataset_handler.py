import os
import kaggle
import pandas as pd
import random
import json

class DatasetHandler:
    def __init__(self, cache_dir='./data'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._setup_kaggle()
    
    def _setup_kaggle(self):
        """Setup Kaggle API credentials."""
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json):
            print("Kaggle credentials not found. Please follow these steps:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll down to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. Save the downloaded kaggle.json file to ~/.kaggle/")
            print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
            raise Exception("Kaggle credentials not found. Please follow the instructions above.")
        
        # Ensure proper permissions
        if os.path.exists(kaggle_json):
            os.chmod(kaggle_json, 0o600)
    
    def load_dataset(self):
        # Check if dataset is already downloaded
        dataset_path = os.path.join(self.cache_dir, 'Reviews.csv')
        if os.path.exists(dataset_path):
            print("Loading dataset from cache...")
            return pd.read_csv(dataset_path)
        else:
            print("Downloading dataset from Kaggle...")
            try:
                # Download the dataset using Kaggle API
                kaggle.api.dataset_download_files(
                    'arhamrumi/amazon-product-reviews',
                    path=self.cache_dir,
                    unzip=True
                )
                # Load the CSV file
                return pd.read_csv(dataset_path)
            except Exception as e:
                print(f"Error downloading dataset: {str(e)}")
                print("Please ensure you have:")
                print("1. A Kaggle account")
                print("2. Accepted the dataset rules at https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews")
                print("3. Properly set up your Kaggle API credentials")
                raise
    
    def get_random_products(self, num_products=10):
        df = self.load_dataset()
        
        # Get unique product IDs
        unique_products = df['ProductId'].unique()
        selected_products = random.sample(list(unique_products), min(num_products, len(unique_products)))
        
        # Get all reviews for selected products
        product_reviews = {}
        for product_id in selected_products:
            reviews = df[df['ProductId'] == product_id]['Text'].tolist()
            product_reviews[product_id] = ' '.join(reviews)
        
        return product_reviews 