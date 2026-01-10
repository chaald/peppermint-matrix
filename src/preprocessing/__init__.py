import pandas as pd
from collections import defaultdict
from typing import Dict

FeatureMeta = Dict[str, Dict[str, object]]

def construct_features_meta(user_interaction_dataframe:pd.DataFrame):
    features_meta = defaultdict(dict)
    for feature_name in user_interaction_dataframe.columns:
        features_meta[feature_name]["dtype"] = user_interaction_dataframe[feature_name].dtype.name
        features_meta[feature_name]["unique_count"] = user_interaction_dataframe[feature_name].nunique()
        features_meta[feature_name]["vocabulary"] = user_interaction_dataframe[feature_name].unique().tolist()

    return dict(features_meta)