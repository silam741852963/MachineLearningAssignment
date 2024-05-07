from utils.export_popular_words import export_popular_words
from utils.load_data import load_data
from utils.extract_features import extract_features
from utils.preprocess_features import preprocess_features
from utils.train_and_evaluate_model import train_and_evaluate_model
from utils.predict_new_data import predict_new_data

#-----------------------Extract features-------------------------------------

# export_popular_words()
# df = load_data()
# df_exploded = extract_features(df)

labeled_data = load_data("processed_data/labeled_data.xlsx")

#-----------------------Preprocess-------------------------------------------

features = preprocess_features(labeled_data)

#-----------------------Train------------------------------------------------

train_and_evaluate_model(features, labeled_data['NER'])

#-----------------------Use--------------------------------------------------

predict_new_data()