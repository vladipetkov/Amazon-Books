# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project Big Data - Notebook Group 22; Amazon Books
#
# ## Datasets Import and Setup
# In the following block we import the datasets and prepare them for further usage.

# %%
import warnings

warnings.filterwarnings("ignore")

import os
import pycountry
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, fbeta_score, make_scorer, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss, classification_report

from sklearn.calibration import CalibrationDisplay

RANDOM_STATE = 45
np.random.seed(RANDOM_STATE)

CUR_DIR = os.getcwd()
BOOKS_DATA_PATH = os.path.join(CUR_DIR, "books_data/books.csv")
RATINGS_DATA_PATH = os.path.join(CUR_DIR, "books_data/ratings.csv")
USERS_DATA_PATH = os.path.join(CUR_DIR, "books_data/users.csv")

df_books = pd.read_csv(BOOKS_DATA_PATH, encoding = 'latin-1', sep = ';', quotechar='"', escapechar="\\").drop(columns = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
df_ratings = pd.read_csv(RATINGS_DATA_PATH, encoding = 'latin-1', sep = ';', quotechar='"')
df_users = pd.read_csv(USERS_DATA_PATH, encoding = 'latin-1', sep = ';', quotechar='"', escapechar="\\", index_col=0)

print(f"Books data    loaded {len(df_books):,} rows, {df_books.shape[1]} columns.")
print(f"Ratings data  loaded {len(df_ratings):,} rows, {df_ratings.shape[1]} columns.")
print(f"Users data    loaded {len(df_users):,} rows, {df_users.shape[1]} columns.")

df_active_countries = df_users['Location'].str.split(',').str[-1].str.strip()
unique_active_countries = df_active_countries.unique()

# %% [markdown]
# In the block below we define a list of countries that will be used further. This is a step of cleaning customers data, so we are left only with realistic and standardized countries where customers originate from. A "realistic and standardized country" is a relative term; primarily it is interpreted as the english name or abbreviation of a country's name.
#
# We use a two-step validation method where we first obtain a list of mutual countries and teritories, present in the dataset and in the python library pycountry, storing official names of most countries in the world (METHOD 1). Then (in METHOD 2) we check the data from countries with more than 50 customers, and compare if we are missing any country from the "mutual" list. We manually add those missing countries to the "mutual" list and then we convert the ones that have two names in the dataframe using a map (example is 'usa' that was manually added due to a big proportion being under that name and 'united states'). 
#
# At the end we preserved about 98% of the users from the original set, organizing them in each country accordingly. The other 2% is removed due to the unstandardized nature of the addresses. Note, there are two countries with more than 50 users, that we decided not to include: "españa" due to having about 65 customers, while the majority has correctly used the standard "spain", and "yugoslavia" due to having about 180 customers, however the country no longer existing.

# %%
"""
METHOD 1
"""
countries = []
for country in pycountry.countries:
    countries.append(country.name.lower())

mutual = []
for mutual_country in countries:
    if mutual_country in unique_active_countries:
        mutual.append(mutual_country)

#ADD MANUALLY countries not in mutual
mutual += "usa", "russia", "u.a.e", "turkey", "españa", "iran", "vietnam", "taiwan", "syria", "venezuela", "south korea", "czech republic"
 
total_users_method1 = 0
for country in mutual:
    total_users_method1 += df_active_countries[df_active_countries == country].count()
print(f"Percentage of users kept after cleaning;            METHOD 1: {round(total_users_method1/len(df_users)*100, 2)}")

"""
METHOD 2
this is an additional validation step of the mutual list
"""
countries_counts = df_active_countries.value_counts()
counts_list = countries_counts[countries_counts>50].index.tolist()
counts_list.remove("")

total_users_method2 = 0
for country in counts_list:
    total_users_method2 += df_active_countries[df_active_countries == country].count()
print(f"Percentage of users kept after cleaning; only using METHOD 2: {round(total_users_method2/len(df_users)*100, 2)}")
print(f"Countries, having more than 50 customers, excluded from the countries list are: {set(counts_list)- set(mutual)}")

df_users['Country'] = df_users['Location'].str.split(',').str[-1].str.strip().str.lower()
df_users['Country'] = df_users['Country'].apply(lambda x: x if x in mutual else "unknown")

countries_map = {'usa' : 'united states',
                 'russian federation' : 'russia',
                 'türkiye' : 'turkey',
                 'u.a.e' : 'united arab emirates',
                 'españa' : 'spain',
                 'hong kong': 'china',
            }
df_users['Country'] = df_users['Country'].replace(countries_map)

df_users = df_users.drop(columns = ['Location'])

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# This section explores the three original datasets before building the recommendation model. The goal is to understand the structure of the data, check missing values, inspect important variables, and identify possible limitations that may affect the model later.

# %% [markdown]
# ### Dataset overview
# This section explores the three original datasets before building the recommendation model.

# %%
datasets = {
    "books": df_books,
    "ratings": df_ratings,
    "users": df_users
}

for name, df in datasets.items():
    print(f"\n{name.upper()} DATASET")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:", df.isnull().sum().sum())

# %% [markdown]
# ### Books Dataset
# We explore the books dataset. We check the number of unique books, missing values, and the most common authors and publishers. This helps us understand which books and publishers are most represented in the data.

# %%
print("Data Types of Books Dataset:")
print("\n", df_books.dtypes)

print("\nNumber of unique books:", df_books["ISBN"].nunique())
print("\nMissing values in books dataset:\n", df_books.isnull().sum())


top_n = 10

author_count = (
    df_books["Book-Author"]
    .value_counts()
    .head(top_n)
    .reset_index()
)
author_count.columns = ["Book-Author", "Count"]

publisher_count = (
    df_books["Publisher"]
    .value_counts()
    .head(top_n)
    .reset_index()
)
publisher_count.columns = ["Publisher", "Count"]

year_count = (
    df_books[df_books["Year-Of-Publication"].notna()]
    ["Year-Of-Publication"]
    .astype(int)
    .value_counts()
    .head(top_n)
    .reset_index()
)
year_count.columns = ["Year-Of-Publication", "Count"]
year_count["Year-Of-Publication"] = year_count["Year-Of-Publication"].astype(str)

most_rated_books = (
    df_ratings
    .groupby("ISBN")
    .size()
    .reset_index(name="Count")
    .merge(df_books[["ISBN", "Book-Title"]].drop_duplicates("ISBN"),on="ISBN",how="left")
    .sort_values("Count", ascending=False)
    .head(top_n)
)

fig, axes = plt.subplots(2, 2, figsize=(22, 12))

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

sns.barplot(data=author_count, y="Book-Author", x="Count", ax=ax1)
ax1.set_title("Top 10 Most Published Authors")
ax1.set_xlabel("Number of Books")
ax1.set_ylabel("Author")

sns.barplot(data=publisher_count, y="Publisher", x="Count", ax=ax2)
ax2.set_title("Top 10 Publishers")
ax2.set_xlabel("Number of Books")
ax2.set_ylabel("Publisher")

sns.barplot(data=year_count, y="Year-Of-Publication", x="Count", ax=ax3)
ax3.set_title("Top 10 Publication Years")
ax3.set_xlabel("Number of Books")
ax3.set_ylabel("Year")

sns.barplot(data=most_rated_books, y="Book-Title", x="Count", ax=ax4)
ax4.set_title("Top 10 Books with Most Ratings")
ax4.set_xlabel("Number of Ratings")
ax4.set_ylabel("Book Title")

for ax in [ax1, ax2, ax3, ax4]:
    ax.bar_label(ax.containers[0])
    ax.margins(x=0.15)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Ratings Dataset
# Here we explore the ratings dataset. We check how many users and books appear in the ratings table and look at the distribution of ratings.

# %%
print("Data Types of Ratings Dataset:")
print("\n", df_ratings.dtypes)

print("\nNumber of unique rated books:", df_ratings["ISBN"].nunique())
print("Number of unique rating users:", df_ratings["User-ID"].nunique())
print("\nMissing values in ratings dataset:\n", df_ratings.isnull().sum())

plt.figure(figsize=(10, 5))
sns.countplot(x="Book-Rating", data=df_ratings)
plt.title("Distribution of Ratings")
plt.xlabel("Book Rating")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ### Users Dataset
# In this section, we explore the users dataset. We check missing values, user information, and the distribution of age.

# %%
print("Data Types of Users Dataset:")
print("\n", df_users.dtypes)

print("\nNumber of unique users:", df_users.index.nunique())
print("\nMissing values in users dataset:\n", df_users.isnull().sum())

plt.figure(figsize=(10, 5))
sns.histplot(df_users[(df_users["Age"] > 6) & (df_users["Age"] < 100)]["Age"].dropna(), bins=25)
plt.title("Distribution of User Age")
plt.xlabel("Age")
plt.ylabel("Number of Users")
plt.show()

# %% [markdown]
# ## Advanced Exploratory Data Analysis (EDA)
# After exploring the three original datasets separately, this section looks deeper into patterns.
#
# We create derived columns and summary metrics. These help us understand how users interact with books, whether certain groups behave differently, and which variables should or should not be used in the final model.
#
# This part includes derived publication-year features, country-level rating metrics, and age-based statistics.

# %% [markdown]
# ### Derived Columns
# This section of code adds new derived columns to the users dataframe based on the Publishing Year column of the books data. The new columns represent the number of books that a user has read from each decade and the average rating the user has given to books published in that decade.

# %%
df_aux = df_ratings.merge(df_books[['ISBN','Year-Of-Publication']], how = 'left', on = 'ISBN')


df_aux = df_aux[df_aux['Year-Of-Publication'] >= 10]
df_aux['Decade-Of-Publication'] = df_aux['Year-Of-Publication'].apply(lambda row:row // 100* 100)
df_aux = df_aux[['User-ID','Decade-Of-Publication','ISBN']].groupby(['User-ID','Decade-Of-Publication']).count()

df_aux = df_aux.pivot_table(
    index = 'User-ID',
    columns= 'Decade-Of-Publication',
    values = 'ISBN',
    fill_value = 0
).reset_index()

df_aux = df_aux.merge(df_users.reset_index().drop(columns=['Location','Country'], errors='ignore'), on='User-ID')
df_aux.corr(method='pearson').style.background_gradient(cmap='coolwarm')

# %% [markdown]
# ### Countries Metrics
# Below we present basic metrics per country, such as the average scores across countries, and the review frequency (as well as the sample size for each country). 
#
# We remind again that 0-scores represent the purchase of a book and leaving a no score, whereas ratings 1-10 show a customers appreciation of the book.
#
# It is important to notice that all countries are included, disregarding the sample size. This leads to countries inaccurate mean ratings due to limited number of observations within the country. We later investigate countries with bigger sample sizes, so we atempt to extract more informed analytics.
#
# Sample Size graph is splitted into two Reviews Frequency graphs, which are mutually exclusive.

# %%
df_ratings_users = df_ratings.merge(df_users, on = 'User-ID').merge(df_books, on = 'ISBN')

df_no_rating = df_ratings_users[df_ratings_users["Book-Rating"] == 0]
df_rated = df_ratings_users[df_ratings_users["Book-Rating"] > 0]

df_mean_ratings_bycountry = df_rated.groupby('Country')['Book-Rating'].mean().reset_index()
df_samples_sizes_countries = df_ratings_users.groupby('Country')['Book-Rating'].count().reset_index()

df_rating_frequency_leaving_reviews = df_rated.groupby('Country')['Book-Rating'].count().reset_index()
df_rating_frequency_not_leaving_reviews = df_no_rating.groupby('Country')['Book-Rating'].count().reset_index()

fig = px.choropleth(
    df_mean_ratings_bycountry,
    locations='Country',
    locationmode='country names',
    color='Book-Rating',
    title='Average Book Rating per Country',
    hover_data='Book-Rating',
    color_continuous_scale='bluyl'
)
fig.show()

fig = px.choropleth(
    df_samples_sizes_countries,
    locations='Country',
    locationmode='country names',
    color='Book-Rating',
    title='Sample Sizes per Country',
    hover_data='Book-Rating',
    color_continuous_scale='bluyl'
)
fig.show()

fig = px.choropleth(
    df_rating_frequency_leaving_reviews,
    locations='Country',
    locationmode='country names',
    color='Book-Rating',
    title='Reviews Frequency per Country (Leaving Reviews)',
    hover_data='Book-Rating',
    color_continuous_scale='bluyl'
)
fig.show()

fig = px.choropleth(
    df_rating_frequency_not_leaving_reviews,
    locations='Country',
    locationmode='country names',
    color='Book-Rating',
    title='Reviews Frequency per Country (Not Leaving Reviews)',
    hover_data='Book-Rating',
    color_continuous_scale='bluyl'
)
fig.show()

# %% [markdown]
# ### Metrics for Countries with more than 100 reviews, having left a review
#
# We decided that 100 reviews might be a good starting point, since under Central Limit Theorem, samples with above 30 observations allow the sample mean to be treated as approximately normally distributed, allowing for more reliable estimate of the means.
# In the first graph we have the mean book ratings per country, where we can see that the majority falls between 7 and 9, with average mean of all of about 7.6.
#
# In the second graph we have computed the standard deviation of each country's rating, where we see that the average st. dev. across all is about 1.8. Considering this information, we get that for a randomly picked review we can say that the score might be roughly between 6 and 9.5 (mean ± std). As one can imagine, on the scale 1-10, this suggests that happy customers may tend to leave reviews more often in comparison with unhappy ones, making it a reasonable hypothesis for investigating a positive bias.

# %%
df_std_ratings = df_rated.groupby('Country')['Book-Rating'].std().reset_index()

df_ratings_frequency_morethan_100 = df_rating_frequency_leaving_reviews[df_rating_frequency_leaving_reviews['Book-Rating']>=100]['Country']

df_mean_ratings_morethan_100 = df_mean_ratings_bycountry[df_mean_ratings_bycountry['Country'].isin(df_ratings_frequency_morethan_100)]
df_std_ratings_morethan_100 = df_std_ratings[df_std_ratings['Country'].isin(df_ratings_frequency_morethan_100)]

average_mean_countries = df_mean_ratings_morethan_100['Book-Rating'].mean()
average_std_countries = df_std_ratings_morethan_100['Book-Rating'].mean()

print(f"Average Mean across countries, having more than 100 reviews, is {round(average_mean_countries, 2)}, Average Standard Deviation is {round(average_std_countries, 2)}.")

fig = px.choropleth(
    df_mean_ratings_morethan_100,
    locations='Country',
    locationmode='country names',
    color='Book-Rating',
    title='Average Book Rating per Country for Countries with more than 100 reviews',
    hover_data='Book-Rating',
    color_continuous_scale='bluyl'
)
fig.show()

fig = px.choropleth(
    df_std_ratings_morethan_100,
    locations='Country',
    locationmode='country names',
    color='Book-Rating',
    title='Standard Deviation Book Rating per Country for Countries with more than 100 reviews',
    hover_data='Book-Rating',
    color_continuous_scale='bluyl'
)
fig.show()

# %% [markdown]
# ### Age Statistics
# Below we have plotted the distribution of all users' ages, and the one of the ages of the users that have left a review for a book. Note that the distribution of the users that have purchased a book and have not left a review is identical as the second graph and therefore we have left it out.
#
# We have also calculated the average review per age category, together with the st. deviation, in hopes to see a distinctive pattern, which does not appear here.

# %%
df_rated["Age-Group"] = pd.cut(df_rated["Age"], bins=[0, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300], 
                                labels=["0-15", "15-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100", "100+"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

df_age = pd.cut(df_users["Age"], bins=[0, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300], 
                                labels=["0-15", "15-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100", "100+"]).value_counts().sort_index()
age_distr_plot = df_age.plot(kind = 'bar', ax = ax1, title = 'Age Distribution', color = 'teal')

df_rated_age = df_rated["Age-Group"].value_counts().sort_index()
age_distr_plot = df_rated_age.plot(kind = 'bar', ax = ax2, title = 'Reviews Age Distribution', color = 'teal')

plt.tight_layout()
plt.show()

# %%
df_age_mean_rating = df_rated.groupby("Age-Group")["Book-Rating"].agg(["mean", "std"])

df_age_mean_rating

# %% [markdown]
# ## Two Step Book Recommendation Model
#
# This section builds the recommendation part of the project. The goal of this model is to create a smaller and more relevant set of book recommendations before using a predictive model later. The recommender is based on shared reading behaviour: if users connected to my input books also read another book, that book may be a relevant recommendation.
#
# The system uses a two-step idea. First, the co-reader model finds possible recommendations based on user behaviour. Later, the regression/classification model can rerank these recommendations using extra features.

# %% [markdown]
# ### Data cleaning and Main table creation
# In this section, the ratings, users, and books datasets are merged into one main table called `df_main`. This table contains the user-book interactions together with book information such as title, author, year of publication, and publisher. The country and location columns are removed from this model because they are highly unbalanced. Most observations come from the USA. If we used country as a feature, later calculations could become questionable because the model would mostly learn patterns from US users.
#
# The data is restricted to the top 10 publishers. This keeps the model focused on a smaller and cleaner subset of the data. Extra count features are also created:
#
# `book-read-count`: how often a book title appears in the cleaned data.
# `author-read-count`: how often books by the same author appear in the cleaned data.
#
# A separate table, `df_book_list`, is created to store one row per ISBN. This table is used later to display book information in the recommendation output.

# %%
#merge all data into one table - df_main
df_main = pd.merge(df_ratings,df_users, how='left', on='User-ID')
df_main = pd.merge(df_main, df_books, how= 'left', on='ISBN')
df_main = df_main.drop(columns=['Location', 'Country'], errors='ignore')
#restricts data only to top publishers
top_publishers = df_main['Publisher'].value_counts().head(10).index.to_list()
df_main = df_main[df_main['Publisher'].isin(top_publishers)]

#creates book-read-count column
read_count = df_main['Book-Title'].value_counts()
read_count.name = 'book-read-count'

#creates author-read-count column
author_counts = df_main['Book-Author'].value_counts()
author_counts.name = 'author-read-count'

#adds author_counts and read_count to df_main
df_main = pd.merge(df_main, author_counts, how='left', on='Book-Author')
df_main = pd.merge(df_main, read_count, how='left', on='Book-Title')

#create df_book_list table for recommendation system
df_book_list = (
    df_main[
        [
            'ISBN',
            'Book-Title',
            'Book-Author',
            'Year-Of-Publication',
            'Publisher',
            'book-read-count',
            'author-read-count'
        ]
    ]
    .drop_duplicates('ISBN')
    .copy()
)


# %% [markdown]
# ### Book ISBN search based on title for future actions
# This section creates a helper function to search books by title. Since the recommendation model uses ISBNs as input, this function helps find the correct ISBN for a book title.

# %%
def search_books_by_title(search_text, n=10):
    """
    Search books by title so you can find the correct ISBN.
    """
    search_text = str(search_text).lower()

    results = df_book_list[
        df_book_list['Book-Title']
        .astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
    ].copy()

    results = results.sort_values(
        by='book-read-count',
        ascending=False
    )

    return results.head(n)


# %%
search_books_by_title("Angels", n=10) 


# %% [markdown]
# ### Co-reader vote recommender

# %% [markdown]
# #### Recommendation model logic
#
# This section contains the main recommendation function, `recommender(seed_isbns, k)`. The function takes a list of input books and recommends other books based on users who are connected to those input books.
#
# The model uses one main logic: find matched users, look at what else they read, and give votes to those other books. The value of k controls how strict the matching rule is.
#
# If k = 0, any interaction with an input book counts as a match. This gives broader recommendations based on general reading behaviour.
#
# If k > 0, a user only counts as matched if they rated an input book at least k. This gives stricter recommendations based on books users actually rated highly.
#
# For each matched user, the model counts how many input books they matched. This count becomes the vote weight. Every other book read by that user receives votes equal to this weight.
#
# For example, suppose the input books are Book A, Book B, and Book C. If User A read all three books, then with k = 0, User A gives 3 votes to every other book they read. If k = 8 and User A only rated Book A above 8, then User A gives only 1 vote. If User B rated Book A and Book B above 8, then User B gives 2 votes to every other book they read.
#
# The recommender returns a ranked list of candidate books. The most important output columns are:
#
# - `coreader_vote_score`: total weighted votes for the candidate book.
# - `matched_reader_count`: number of matched users who read the candidate book.
# book information such as title, author, year, and publisher.

# %%
def recommender(seed_isbns, k, df_main):
    # Clean input ISBNs
    seed_isbns = [str(isbn).strip() for isbn in seed_isbns]
    seed_isbns = list(set(seed_isbns))

    # Read/interacted events from df_main
    read_events = df_main[["User-ID", "ISBN"]].drop_duplicates()

    # Keep only seed books that exist in df_main
    valid_seed_isbns = [
        isbn for isbn in seed_isbns
        if isbn in set(df_main["ISBN"])
    ]

    print("Input seed ISBNs:", seed_isbns)
    print("Valid seed ISBNs:", valid_seed_isbns)

    if len(valid_seed_isbns) == 0:
        raise ValueError("None of the seed ISBNs exist in df_main.")

    seed_ratings = df_main[(df_main["ISBN"].isin(valid_seed_isbns)) & (df_main["Book-Rating"] >= k)].copy()

    if seed_ratings.empty:
        raise ValueError("No users rated the seed books above or equal to k.")

    user_seed_matches = (seed_ratings.groupby("User-ID")["ISBN"].nunique())
    user_seed_matches.name = "liked_seed_count"
    user_seed_matches = user_seed_matches.reset_index()

    score_column = "liked_seed_count"
    final_score_name = "vote_score"
    print("Matched users:", len(user_seed_matches))



    # Get all other books read by matched users
    candidate_reads = pd.merge(read_events, user_seed_matches, how="inner", on="User-ID")

    # Do not recommend the input seed books
    candidate_reads = candidate_reads[~candidate_reads["ISBN"].isin(valid_seed_isbns)].copy()
    if candidate_reads.empty:
        return pd.DataFrame()

    # Score candidate books
    scores = (
        candidate_reads
        .groupby("ISBN")
        .agg(
            vote_score=(score_column, "sum"),
            matched_reader_count=("User-ID", "nunique")
        ).reset_index())
    scores = scores.rename(columns={"vote_score": final_score_name})

    # Global reader count
    

    
    scores = scores.sort_values(by=[final_score_name, "matched_reader_count"], ascending=False)

    recommendations = pd.merge(scores, df_book_list, how="left", on="ISBN")

    return recommendations

# %% [markdown]
# ### Recommender Test
# This section tests the recommender with a selected list of seed books. The same input books are tested with different values of k.
#
# This comparison helps show how the threshold k changes the behaviour of the recommender.

# %%
seed_isbns = [
    "0451169530",   # The Stand
    "0451157443",   # Carrie
    "0743424425	",  # The Shining
    "0451184963",   # Insomnia
    "0345337662",   # Interview with the Vampire
    "0440295815",   # Hannibal
    "0399506438",   # Lord of the Flies
    "0345348109"    # The Killer Angels

]

# %%
recommender(seed_isbns, k=0, df_main = df_main)

# %%
recommender(seed_isbns,k=8, df_main = df_main).head()

# %% [markdown]
# ### Extra Data cleaning for Regression
# This code creates the derived columns book-rate-count and book-rate-count, and fills the missing age values with the median. 

# %%
#creates book-rate-count column
df_main_before_mods = df_main.copy()

df_main = df_main[df_main['Book-Rating'] != 0]
rate_count = df_main['Book-Title'].value_counts()
rate_count.name = 'book-rate-count'

#adds rate_count to df_main
df_main = pd.merge(df_main.copy(), rate_count, how= 'left', on='Book-Title')

median_age = df_main['Age'].median()
df_main['Age'] = df_main['Age'].fillna(median_age)
K = 8
df_main['rating-over-k'] = df_main['Book-Rating'].apply(lambda row: 1 if row >= K else 0)

display(df_main)

NUM_COLS = [
    'book-rate-count',
    'book-read-count',
    'author-read-count',
    'Year-Of-Publication',
    'Age',
    'vote_score',
    'matched_reader_count'
]
CAT_COLS = [
    'Publisher',
]
TARGET = 'rating-over-k'


# %% [markdown]
# ### Model testing preparation
# This cell prepares some useful functions for creating preprocessors and pipelines for the different classification models, creates the StratifiedKFold() object used for cross validating later and splits the data into a training set and a test set so that we can provide an honest final answer, with no leakege.

# %%
def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ]
    )

def make_pipeline_for(model_name: str, **params) -> Pipeline:
    if model_name == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)
        scale_numeric = True
    elif model_name == "tree":
        estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        scale_numeric = False
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
        scale_numeric = False
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        scale_numeric = False
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        [
            ("pre", make_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )

# Base Cross-Validation Strategy
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)



df_test = pd.merge(df_main.copy(), recommender(seed_isbns,k=K, df_main= df_main_before_mods)[['ISBN','vote_score','matched_reader_count']], how = 'left', on = 'ISBN').dropna()

display(df_test)
X = df_test[NUM_COLS + CAT_COLS]
y = df_test[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.8)

# %% [markdown]
# ### Best parameters + Candidates
# This code cell runs a grid search for Random Forest, Gradient boosting and the decision tree, using stratified K-Fold cross validation and maximizing the negative log loss (reason we picked this over f1 is in the report) and then saves and prints the best parameters.
# Aditionally we set up a dictionary with the model names and corresponding pipelines we want to test so that we can iterate over them later. 

# %%
rf_grid = GridSearchCV(
    make_pipeline_for("random_forest"),
    param_grid={
        "model__n_estimators": [100],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 5],
    },
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring="neg_log_loss",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_best_params = {k.replace("model__", ""): v for k, v in rf_grid.best_params_.items()}
print(f"Best RF Params: {rf_best_params}")

# Gradient Boosting GridSearch
gb_grid = GridSearchCV(
    make_pipeline_for("gradient_boosting"),
    param_grid={
        "model__n_estimators": [60, 100],
        "model__learning_rate": [0.05, 0.10],
        "model__max_depth": [2, 3]
    },
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring="neg_log_loss",
    n_jobs=-1
)
gb_grid.fit(X_train, y_train)

gb_best_params = {k.replace("model__", ""): v for k, v in gb_grid.best_params_.items()}
print(f"Best GB Params:{gb_best_params}")

# Decision Tree GridSearch
dt_grid = GridSearchCV(
    make_pipeline_for("tree"),
    param_grid={
        "model__max_depth": [4, 6, 8, 10, None],
        "model__min_samples_leaf": [1, 10, 50, 100]
    },
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring="neg_log_loss",
    n_jobs=-1
)
dt_grid.fit(X_train, y_train)
dt_best_params = {k.replace("model__", ""): v for k, v in dt_grid.best_params_.items()}

print(f"Best DT Params: {dt_best_params}")

# CANDIDATES
candidate_models = {
    "Logistic Regression": make_pipeline_for("logistic"),
    "Shallow Tree": make_pipeline_for("tree", max_depth=4, min_samples_leaf=100),
    "Tuned Decision Tree": make_pipeline_for("tree", **dt_best_params),
    "Random Forest": make_pipeline_for("random_forest", **rf_best_params),
    "Gradient Boosting": make_pipeline_for("gradient_boosting", **gb_best_params)
}

# %% [markdown]
# ### Model score comparison and plot prep
# Inside a for loop iterating through all the models we:
# - prepare some dictionaries and lists that will store data about our models metrics, 
# - create an f-beta scorer inside the for loop so we can calculate the f-beta in each fold (f-Beta is more sensitive to falso positives which is good).
# - collect the out of fold predicted probabilities from the cross validation in oof_predictions_dict (the key is the model name, values is a list of oof probability predictions for the positive class), 
# - calculate f-beta and log loss for each fold and store the individual results of each model in fold_metrics_list (list of dictionaries)
# - calculate the precision and recall for different thresholds using precision_recall_curve()
# - we use those metrics to compute the f-beta
# - we use argmax to find the index of the biggest f-beta value and we save in model_results (a list of dictionaries), the model name, optimal threshold and maximal f-beta
#
# After the for loop we plot the F-Beta scores of each model agains the corresponding threshold, and display a "scorecard" of what the best threshold and F-beta was for each model. In the end we save the model name and threshold for easier plot making, save the pipeline with the highest F-Beta in winning_pipeline and we turn the fold metrics into a dataframe for easier ploting

# %%
print("Generating OOF predictions and metrics...")

oof_predictions_dict = {}
model_results = []
fold_metrics_list = []

plt.figure(figsize=(10, 6))

for name, pipeline in candidate_models.items():
    # Get out of fold probabilities
    oof_probs =cross_val_predict(
        pipeline,X_train, y_train, cv=cv5, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_predictions_dict[name] = oof_probs
    
    # Calculate oof metrics
    f05_scorer = make_scorer(fbeta_score, beta=0.5)
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv5, scoring={'fbeta': f05_scorer, 'log_loss': 'neg_log_loss'})

    for fold_idx in range(cv5.get_n_splits()):
        fold_metrics_list.append({
            "Model": name,
            "Fold":fold_idx +1,
            "F0.5 Score (Default Threshold 0.5)": cv_scores['test_fbeta'][fold_idx],
            "Neg Log-Loss": cv_scores['test_log_loss'][fold_idx]
        })
    
    # calculate best threshold
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)

    beta = 0.5
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * (precisions * recalls)
    denominator = (beta_sq * precisions) + recalls
    fbeta_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    best_idx = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 =  fbeta_scores[best_idx]
    
    model_results.append({
        "Model": name,
        "Optimal Threshold": best_threshold,
        "Peak OOF F05-Score": best_f1
    })
    
    # Plot fbeta values
    plt.plot(thresholds, fbeta_scores[:-1], label=f"{name} (Peak: {best_f1:.3f})", linewidth=2)

# Threshold vs Fbeta Plot
plt.title("OOF F05-Score vs. Decision Threshold", fontsize=14)
plt.xlabel("Probability Threshold")
plt.ylabel("F05 Score")
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label="Default 0.5 Threshold")
plt.legend(loc="lower center")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary Scorecard
summary_df = pd.DataFrame(model_results).sort_values(by="Peak OOF F05-Score", ascending=False)
display(summary_df.round(4).reset_index(drop=True))

# Get winning details
best_model_name = summary_df.iloc[0]["Model"]
best_threshold = summary_df.iloc[0]["Optimal Threshold"]
winning_pipeline = candidate_models[best_model_name]

# CV metrics df
cv_metrics_df = pd.DataFrame(fold_metrics_list)


# %% [markdown]
# ### Cross Validation Fold Metrics
# In this section we combine a swarmplot and boxplot to show the variation and distribution of the negative log-loss and F-Beta from each fold in the cross-validation performed above

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: FBeta Score (Default threshold) variance
sns.boxplot(data=cv_metrics_df , x="Model", y="F0.5 Score (Default Threshold 0.5)", color="white", ax=axes[0])
sns.swarmplot(data=cv_metrics_df, x="Model", y="F0.5 Score (Default Threshold 0.5)",  size=8, ax=axes[0])
axes[0].set_title("CV F05-Score Stability (Default Threshold)", fontsize=13)
axes[0].set_ylabel("F05 Score")

# Subplot 2: Negative Log-Loss variance
sns.boxplot(data=cv_metrics_df, x="Model", y="Neg Log-Loss", color="white", ax=axes[1])
sns.swarmplot(data=cv_metrics_df, x="Model", y="Neg Log-Loss", size=8, ax=axes[1])
axes[1].set_title("CV Neg Log-Loss Stability" , fontsize=13)
axes[1].set_ylabel("Negative Log-Loss (Higher is Better)")

plt.suptitle("Cross-Validation Fold Performance Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature importances/coefficients
# This cell fits all the models on the full training set and extracts the coefficients in the case of Logistic Regression and the feature importances in the case of the other methods and plots all of them by model, ordered descendingly.

# %%
#  Feature Importances

fig, axes =plt.subplots(1, len(candidate_models), figsize=(18, 6))

for idx,(name, pipeline) in enumerate(candidate_models.items()):
    # fit pipeline
    pipeline.fit(X_train, y_train)
    
    # get freatures
    preprocessor = pipeline.named_steps['pre']
    model = pipeline.named_steps['model']
    feature_names_out = preprocessor.get_feature_names_out()
    
    # get importances
    if hasattr(model, 'coef_'): importances = np.abs(model.coef_[0])
    else:
        importances = model.feature_importances_
        
    # plot top 10
    feat_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
    feat_df =feat_df.sort_values(by='Importance', ascending=False).head(10)
    
    sns.barplot(data=feat_df, x='Importance' , y='Feature', ax=axes[idx], palette="viridis")
    axes[idx].set_title(f"{name} Top 10 Features")
    axes[idx].set_xlabel( "Importance (Absolute Coef / Gini)" )
    axes[idx].set_ylabel("")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Model reliability
# This code plots the Calibration Curves of the models so that we can see if they show a pattern of overestimating or underestimating certain predicted probabilities.

# %%
# model reliability

plt.figure(figsize=(8, 8))
ax = plt.gca()

for name in candidate_models.keys():
    CalibrationDisplay.from_predictions(
        y_train, 
        oof_predictions_dict[name], 
        n_bins=10, 
        name=name, 
        ax=ax
    )

plt.title("Calibration Curves (Reliability Diagram)", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Confusion matrix before and after threshold tuning
# This plot is designed to illustrate how the tuned threshould changes the number of false alarms compared to using the default threshold (in our case the two thresholds seem to be almost identical)

# %%
# The Winning Model an Tuned Confusion Matrix
# optimized threshold reditributes errors for the winning model

# Extract winning OOF probabilities
winning_oof_probs = oof_predictions_dict[best_model_name]

# Predict using default 0.5 threshold
y_pred_default = (winning_oof_probs >= 0.5).astype(int)

# Predict using tuned threshold
y_pred_tuned = (winning_oof_probs >= best_threshold).astype(int)

# we plot both
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_default = confusion_matrix(y_train, y_pred_default)
ConfusionMatrixDisplay(cm_default, display_labels=["Negative", "Positive"]).plot(
    ax=axes[0], cmap='Blues',colorbar=False,
    values_format='d'
)
axes[0].set_title(f"{best_model_name} (Default Threshold: 0.50)", fontsize=13)
axes[0].grid(False)

cm_tuned = confusion_matrix(y_train, y_pred_tuned)
ConfusionMatrixDisplay(cm_tuned, display_labels=["Negative", "Positive"]).plot(
    ax=axes[1], cmap='Greens', colorbar=False, values_format='d'
)
axes[1].set_title(f"{best_model_name} (Tuned Threshold: {best_threshold:.3f})", fontsize=13)
axes[1].grid(False)

plt.suptitle("Impact of Threshold Tuning on Training Classification", fontsize=16)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### PR-Curve
# This code plots precision against recall showing how the model precision decreases as it tries to find a bigger share of true positives (recall)

# %%
plt.figure(figsize=(10, 8))
ax = plt.gca()

for name in candidate_models.keys():
    # Plot the PR curve using OOF probs
    PrecisionRecallDisplay.from_predictions(
        y_train,  oof_predictions_dict[name], 
        name=name, 
        ax=ax,
        linewidth=2
    )



plt.title("Precision-Recall Curve (OOF Predictions)", fontsize=14)
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Final model summary + Confusion Matrix
# This final section calculates a variety of metrics for the winning model, and a final confusion matrix. The model is fitted on the held out part of the data specifically intended for reporting. The confusion matrix shows the final ammount of False alarms and correct predictions.

# %%
print(f"TEST SET EVALUATION")
print(f"Model: {best_model_name}")
print(f"Threshold: {best_threshold:.3f}\n")

# get probs of test set
test_probs = winning_pipeline.predict_proba(X_test)[:, 1]

# apply best threshold
test_preds = (test_probs >= best_threshold).astype(int)

# Calculate metrics
print("METRICS")
print(f"Accuracy:      {accuracy_score(y_test, test_preds):.4f}")
print(f"Precision:     {precision_score(y_test, test_preds):.4f}")
print(f"Recall:       {recall_score(y_test, test_preds):.4f}")
print(f"F1 Score:      {f1_score(y_test, test_preds):.4f}")
print(f"F0.5 Score:    {fbeta_score(y_test, test_preds, beta=0.5):.4f}")

print("\nPROBABILITY METRICS")
print(f"ROC AUC:       {roc_auc_score(y_test, test_probs):.4f}")
print(f"Log Loss:      {log_loss(y_test, test_probs):.4f}")
print(f"Brier Score:   {brier_score_loss(y_test, test_probs):.4f}")

print( "\nCLASSIFICATION REPORT")
print(classification_report(y_test, test_preds,target_names=["Negative", "Positive"]))

#Plot the Final Confusion Matrix
plt.figure(figsize=(7,6))
cm_test = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(cm_test, display_labels=["Negative", "Positive"])
disp.plot(cmap='Purples', colorbar=False, values_format='d',ax=plt.gca())
plt.title(f"Final Test Set Confusion Matrix\n({best_model_name} @ {best_threshold:.3f})", fontsize=14)
plt.show()

# %%
print("This would be the accuracy if every book was predicted as a 1:", df_main['rating-over-k'].mean())

# %% [markdown]
# ### Revisiting the recommender
# Here we take the average predicted probability of the top 10 books from the recommender in order to see how to vote_score and average predicted prob contrast eachother.

# %%
column_names = X_test.columns
df_recommended_books = pd.concat([X_test.reset_index(), pd.DataFrame(test_probs, columns = ['pred_prob'])], axis = 1)
df_recommended_books.index = df_recommended_books['index']
#df_recommended_books.columns = column_names+['pred_prob']
print(column_names)
df_recommended_books = df_recommended_books.join(df_main[['ISBN','Book-Title']], how = 'left')
df_recommended_books = df_recommended_books[['pred_prob', 'vote_score', 'Book-Title']]

avg_proba_df = df_recommended_books.groupby('Book-Title')['pred_prob'].mean().reset_index()

final_table = avg_proba_df.merge(df_recommended_books[['Book-Title', 'vote_score']].drop_duplicates(), on='Book-Title', how='left')
final_table = final_table[['vote_score', 'pred_prob', 'Book-Title']]
final_table = final_table.sort_values(by='pred_prob', ascending=False)

display(final_table.sort_values(by = 'vote_score', ascending = False).head(10))
