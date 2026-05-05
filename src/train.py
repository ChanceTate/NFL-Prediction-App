from data import load_data
from features import build_features

df = load_data()

X = build_features(df)