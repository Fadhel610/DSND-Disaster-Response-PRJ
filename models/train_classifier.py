import sys
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    '''

    A fucntion that load the data from a sqlite database.
    Input:
        database_filepath: The database filepath
    Return:
        X: The messages
        Y: The category of the message
        category_names: The lables for 36 categories

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''

    A function that tokenize and clean an input text
    Input:
        text: The original text message
    Return:
        clean_tokens: Tokenized and cleaned text

    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    
    A class tht add a feature to the ML model by extracting the first verb
    
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''

    A function that builds the model through pipeline
    Return:
        model: a model that that processes text messages and classifies them

    '''
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''

    A function that evaluate the model
    Input:
        model: The trained model to be evaluated
        X_test: The input messages that will test the model
        Y_text: The correct output that the model will be compared to
        category_names: Labels
    Output:
        The function will print the accuracy of the model for each lable and print the detailed reprot in a text file
    
    '''
    Y_pred = model.predict(X_test)
    
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('-> The overall accuracy is: ', 100*overall_accuracy)
    file = open('Results_Report', 'w+')
    file.write('This file contains the Classification Rerpot\n\n')
    file.write('-> Accuracy for each column is:')
    file.write(str((Y_pred == Y_test).mean()*100))
    file.write('\n\n ---- The Detailed Reprot ---')
    y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
   # file = open('Results_Report', 'w+')
    for col in Y_test.columns:
        file.write('\n -> Category: {}'.format(col.upper()))
        file.write(classification_report(Y_test[col], Y_pred_df[col]))
    file.close()
    pass


def save_model(model, model_filepath):
    '''

    A function that saves the model as pkl to be used later
    Input:
        model: The model to be saved
        model_filepath:  The file path where the model will be saved

    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    '''
    The main function that will:
    1. Get the data from sqlite database
    2. Train a ML model to classify the text messages
    3. Evaluate the model
    4. Save the trained model as pkl

    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
