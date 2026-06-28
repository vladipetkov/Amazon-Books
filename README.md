# Amazon Books Project
This repository contains coding and documentation supporting the underdeveloped project for the course Big Data at Vrije Universiteit Amsterdam.

## Members
Colaborators in this project are Vladimir Petkov, Mark Nemeth, Vlad Mare and Alan Chen, all members of Group 22.

## Data Source
The dataset was obtained via kaggle and could be found in the following link:
https://www.kaggle.com/datasets/saurabhbagchi/books-dataset

It contains records for available books and users, together with some of their properties respectively, and ratings made by users for book purchases.
It is worth mentioning that the ratings table contains records for purchases and ratings altogether. Therefore one could separate the purchased, but non-rated reviews, having a rating of 0, from the actual ratings, measured on the scale 1-10.

## Research Questions
The group explored the following research questions in their work, amongst many others.

### How can user reading behavior be used to recommend books to readers in the Amazon Books dataset?

### Which book and user-related features help predict whether a book will receive a high rating?

### Can we use user- and book-related features to predict whether a book will receive a high rating?

## Instructions for running the files
The notebook file has already the produced outputs and results displayed. If one wishes to run the files they would need to have installed the appropriate packages, mentioned in the requirements.txt
Required packages are:
pycountry
plotly
numpy
pandas
matplotlib
seaborn
ipython
scikit-learn
