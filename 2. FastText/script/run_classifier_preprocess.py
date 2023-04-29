import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

import pickle
from src.preprocess import preprocess_TC
import argparse

datum = ("ag", "amazon_f", "amazon_p", "dbpedia", "sogou", "yahoo", "yelp_f", "yelp_p")

for data in ['sogou']:
    print(f"preprocess for {data.upper()}...")
    data_out = preprocess_TC(data)

    with open(data + ".pkl", 'wb') as f:
        pickle.dump(data_out, f)