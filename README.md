# DSND-Disaster-Response-PRJ
A project from Udacity's DSND aims at creating pipeline that processes messages from a web app
# Installation
It is required to have
* Python 3
* Anaconda Distribution; (Pandas, Numpy, sklearn ...)
* wordnet
* stopwords
* punkt

# Motivation
This project is to apply the pipeline concept in processing messages and classifying them.

# File Descriptions
* data
  * process_data.py: This python code takes the raw csv data and clean and save them into sqlite database
  * disaster_category.csv: This is the data that have all the categories of the messages
  * disaster_messages.csv: This is the data that have all the messages
* models
  * train_classifier.py: This python code takes the data from the database and train the ML model
* app
  - run.py: This python code runs the web application

### Upon running the scripts, new files will appear, which are
  * data/DisasterResponse.db: This is the created database
  * models/classifier.pkl: This is the trained model

# Instructions to Execute the Program
First you need to run `process_data.py`, then `train_classifer.py` and finally `run.py`.
After that, go to http://0.0.0.0:3001 to interact with the web app.
You will see two visuals related to the used data and you will be able to type a message and then see how the model classification 
