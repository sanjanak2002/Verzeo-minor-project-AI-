# main.py

import pandas as pd
from preprocessing import TextPreprocessor
from model import ToxicCommentsModel

if __name__ == '__main__':
    # Load data
    train = pd.read_csv('../input/train.csv').fillna(' ')
    test = pd.read_csv('../input/test.csv').fillna(' ')

    # Preprocess data
    preprocessor = TextPreprocessor()
    preprocessor.get_indicators_and_clean_comments(train)
    preprocessor.get_indicators_and_clean_comments(test)

    # Feature engineering and model training/prediction
    model = ToxicCommentsModel()
    model.feature_engineering(train, test)
    model.train_and_predict(train, test, class_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
