from sklearn.metrics import classification_report
from export_popular_words import export_popular_words
from load_training_data import load_training_data
from extract_features import extract_features
from preprocess_features import preprocess_features
from train_and_evaluate_model import train_and_evaluate_model

#-----------------------Extract features-------------------------------------

export_popular_words()
df = load_training_data()
df_exploded = extract_features(df)

#-----------------------Preprocess-------------------------------------------

features = preprocess_features(df_exploded)

#-----------------------Train-------------------------------------

train_and_evaluate_model(features, df_exploded['is_chunk'])