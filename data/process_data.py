import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''

    This function load the data and merge them then return a dataframe
    Inputs:
        messages_filepath: The path to the messages file
        categories_filepath: The path to the categories file
    Return:
        df: A dataframe that contains both messages and categories merged.

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    '''

    This function takes a dataframe and clean it
    Inputs:
        df: A raw dataframe
    Return:
        df: A cleaned dataframe

    '''
    
    categories = df['categories'].str.split(';',expand = True )
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df.drop(columns='categories', inplace = True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''

    A function that save the dataframe to a sqlite database
    Inputs:
        df: A dataframe
        database_filename: The name of the database

    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)
    pass  


def main():
    '''
    The main function.
    1. Extracts the raw data
    2. Cleans the data
    3. Save the data to the a sqlite database

    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
