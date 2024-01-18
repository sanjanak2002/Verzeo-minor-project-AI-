# model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from collections import defaultdict
import numpy as np
import pandas as pd

class ToxicCommentsModel:
    def __init__(self):
        # Initializing model parameters
        self.svm_params = {
            'C': 4.0,
            'penalty': 'l1',
            'dual': False,
            'random_state': 1,
            'max_iter': 100
        }

    def feature_engineering(self, train, test):
        # Scaling numerical features with MinMaxScaler
        num_features = [f_ for f_ in train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address'] + class_names]

        scaler = MinMaxScaler()
        train_num_features = csr_matrix(scaler.fit_transform(train[num_features]))
        test_num_features = csr_matrix(scaler.transform(test[num_features]))

        # Get TF-IDF features
        train_text = train['clean_comment']
        test_text = test['clean_comment']
        all_text = pd.concat([train_text, test_text])

        # First on real words
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)
        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features = word_vectorizer.transform(test_text)

        del word_vectorizer
        gc.collect()

        # Now using the char_analyzer to get another TFIDF
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 1),
            max_features=50000)
        char_vectorizer.fit(all_text)
        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)

        del char_vectorizer
        gc.collect()

        print((train_char_features > 0).sum(axis=1).max())

        del train_text
        del test_text
        gc.collect()

        # Now stacking TF IDF matrices
        csr_trn = hstack(
            [
                train_char_features,
                train_word_features,
                train_num_features
            ]
        ).tocsr()
        del train_num_features
        del train_char_features
        gc.collect()

        csr_sub = hstack(
            [
                test_char_features,
                test_word_features,
                test_num_features
            ]
        ).tocsr()
        del test_num_features
        del test_char_features
        gc.collect()

        return csr_trn, csr_sub

    def train_and_predict(self, train, test, class_names):
        # Set SVM parameters
        svm_params = {
            'C': 4.0,
            'penalty': 'l1',
            'dual': False,
            'random_state': 1,
            'max_iter': 100
        }

        # Initializing K-Fold
        folds = KFold(n_splits=4, shuffle=True, random_state=1)
        scores = []

        # Initializing SVM model
        svm_model = LinearSVC(**svm_params)

        for class_name in class_names:
            print("Class %s scores: " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]

            # Feature selection using linear SVM
            selector = SelectFromModel(estimator=svm_model)
            selector.fit(csr_trn, train_target)
            csr_trn_selected = selector.transform(csr_trn)
            csr_sub_selected = selector.transform(csr_sub)

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                # Training SVM
                svm_model.fit(csr_trn_selected[trn_idx], train_target.values[trn_idx])
                class_pred[val_idx] = svm_model.decision_function(csr_trn_selected[val_idx])

            score = roc_auc_score(train_target.values, class_pred)
            print("\t Average Score: %.6f" % score)
            scores.append(score)

            # Predict probabilities for test data
            test[class_name] = svm_model.decision_function(csr_sub_selected)

        print('Total CV score is {}'.format(np.mean(scores)))

        # Save predictions
        test[['id'] + class_names].to_csv("svm_submission.csv", index=False, float_format="%.8f")
