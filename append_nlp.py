import json

notebook_path = '/Users/baris/Projects/halilinodevi/odev.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Cells to append
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Customer Review Analysis using NLP\n",
            "\n",
            "In this section, we transition from analyzing numerical purchase behavior (RFM) to analyzing unstructured textual data. We will analyze a dataset of Amazon customer reviews to extract sentiment and key textual insights."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 1: Data Loading & Inspection\n",
            "\n",
            "We load a sample of the Amazon Consumer Reviews dataset. We identify `reviews.text` as our main text column and `reviews.rating` as our sentiment proxy."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "\n",
            "# Load a subset of the dataset to keep execution fast\n",
            "dataset_path = '/Users/baris/.cache/kagglehub/datasets/datafiniti/consumer-reviews-of-amazon-products/versions/5/1429_1.csv'\n",
            "df_reviews = pd.read_csv(dataset_path, nrows=5000, low_memory=False)\n",
            "\n",
            "print(\"First 5 rows:\")\n",
            "display(df_reviews[['reviews.rating', 'reviews.text']].head())\n",
            "\n",
            "print(\"\\nColumn names:\", df_reviews.columns.tolist())\n",
            "\n",
            "print(\"\\nMissing values in key columns:\")\n",
            "print(df_reviews[['reviews.rating', 'reviews.text']].isnull().sum())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 2: Text Cleaning\n",
            "\n",
            "We build a preprocessing pipeline using NLTK to:\n",
            "1. Convert text to lowercase\n",
            "2. Remove punctuation and numbers\n",
            "3. Tokenize\n",
            "4. Remove stopwords\n",
            "5. Lemmatize words to their base form"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import re\n",
            "import nltk\n",
            "from nltk.corpus import stopwords\n",
            "from nltk.tokenize import word_tokenize\n",
            "from nltk.stem import WordNetLemmatizer\n",
            "\n",
            "# Ensure NLTK resources are downloaded (handled in background previously, but good practice to include)\n",
            "try:\n",
            "    nltk.data.find('corpora/stopwords')\n",
            "except LookupError:\n",
            "    nltk.download('stopwords')\n",
            "    nltk.download('punkt')\n",
            "    nltk.download('wordnet')\n",
            "\n",
            "stop_words = set(stopwords.words('english'))\n",
            "lemmatizer = WordNetLemmatizer()\n",
            "\n",
            "def clean_text(text):\n",
            "    if not isinstance(text, str):\n",
            "        return \"\"\n",
            "    # 1. Lowercase\n",
            "    text = text.lower()\n",
            "    # 2. Remove punctuation and numbers\n",
            "    text = re.sub(r'[^a-z\\s]', '', text)\n",
            "    # 3. Tokenize\n",
            "    tokens = word_tokenize(text)\n",
            "    # 4 & 5. Remove stopwords and lemmatize\n",
            "    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]\n",
            "    return \" \".join(cleaned_tokens)\n",
            "\n",
            "# Drop any rows without text or rating\n",
            "df_reviews = df_reviews.dropna(subset=['reviews.text', 'reviews.rating']).copy()\n",
            "\n",
            "# Apply cleaning\n",
            "df_reviews['cleaned_text'] = df_reviews['reviews.text'].apply(clean_text)\n",
            "\n",
            "print(\"Sample of cleaned text:\")\n",
            "display(df_reviews[['reviews.text', 'cleaned_text']].head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 3: Text Vectorization\n",
            "\n",
            "Machine learning models require numerical input. We convert our cleaned text into mathematical representations using two common approaches: Bag of Words (BoW) and TF-IDF."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
            "\n",
            "# 3.1 Bag of Words (CountVectorizer)\n",
            "bow_vectorizer = CountVectorizer(max_features=1000)\n",
            "bow_matrix = bow_vectorizer.fit_transform(df_reviews['cleaned_text'])\n",
            "\n",
            "# 3.2 TF-IDF (Term Frequency-Inverse Document Frequency)\n",
            "tfidf_vectorizer = TfidfVectorizer(max_features=1000)\n",
            "tfidf_matrix = tfidf_vectorizer.fit_transform(df_reviews['cleaned_text'])\n",
            "\n",
            "print(f\"BoW Matrix Shape: {bow_matrix.shape}\")\n",
            "print(f\"TF-IDF Matrix Shape: {tfidf_matrix.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 4: Comparison of Methods\n",
            "\n",
            "- **Dimensionality**: Both methods resulted in a matrix of `N x 1000` (since we restricted `max_features=1000` to capture the most important words and save memory).\n",
            "- **Importance Weighting**: \n",
            "  - **Bag of Words (BoW)** simply counts the frequency of a word in a document. A common word like \"tablet\" will have a huge score, even if it doesn't provide unique sentiment value.\n",
            "  - **TF-IDF** scales word frequency by how rare the word is across all documents. It penalizes overly common words and rewards unique, descriptive words.\n",
            "- **Verdict**: **TF-IDF is generally better** for tasks like sentiment analysis because it highlights words that are distinct to a specific review (like \"amazing\" or \"terrible\") rather than words that appear everywhere (like \"amazon\" or \"product\")."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 5: Sentiment Analysis\n",
            "\n",
            "Since our dataset has a `reviews.rating` (1-5), we can create sentiment labels and train a classifier.\n",
            "- **Positive**: Rating > 3\n",
            "- **Negative**: Rating <= 3"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, classification_report\n",
            "\n",
            "# Create Binary Sentiment Label\n",
            "df_reviews['sentiment'] = df_reviews['reviews.rating'].apply(lambda x: 1 if x > 3 else 0)\n",
            "\n",
            "# Train-Test Split using TF-IDF\n",
            "X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df_reviews['sentiment'], test_size=0.2, random_state=42)\n",
            "\n",
            "# Train Logistic Regression Classifier\n",
            "lr_model = LogisticRegression(max_iter=1000)\n",
            "lr_model.fit(X_train, y_train)\n",
            "\n",
            "# Evaluate\n",
            "y_pred = lr_model.predict(X_test)\n",
            "acc = accuracy_score(y_test, y_pred)\n",
            "\n",
            "print(f\"Sentiment Classification Accuracy: {acc * 100:.2f}%\")\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(y_test, y_pred))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 6: Insight Generation\n",
            "\n",
            "We analyze the Bag of Words matrix to find the most frequent words used in strictly Positive vs strictly Negative reviews."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import numpy as np\n",
            "\n",
            "# Separate positive and negative reviews\n",
            "pos_idx = df_reviews[df_reviews['sentiment'] == 1].index\n",
            "neg_idx = df_reviews[df_reviews['sentiment'] == 0].index\n",
            "\n",
            "# Reset index of dataframe to match matrix rows if needed, \n",
            "# but since we used fit_transform on the whole series, rows align.\n",
            "# Sum the word counts for positive and negative subsets\n",
            "pos_word_counts = bow_matrix[df_reviews['sentiment'] == 1].sum(axis=0)\n",
            "neg_word_counts = bow_matrix[df_reviews['sentiment'] == 0].sum(axis=0)\n",
            "\n",
            "# Get feature names (words)\n",
            "words = bow_vectorizer.get_feature_names_out()\n",
            "\n",
            "# Create dataframes for top words\n",
            "pos_freq = pd.DataFrame({'word': words, 'count': np.asarray(pos_word_counts).flatten()})\n",
            "neg_freq = pd.DataFrame({'word': words, 'count': np.asarray(neg_word_counts).flatten()})\n",
            "\n",
            "top_pos = pos_freq.sort_values(by='count', ascending=False).head(10)\n",
            "top_neg = neg_freq.sort_values(by='count', ascending=False).head(10)\n",
            "\n",
            "# Plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
            "\n",
            "sns.barplot(data=top_pos, x='count', y='word', ax=axes[0], palette='Greens_r')\n",
            "axes[0].set_title('Top 10 Words in Positive Reviews')\n",
            "\n",
            "sns.barplot(data=top_neg, x='count', y='word', ax=axes[1], palette='Reds_r')\n",
            "axes[1].set_title('Top 10 Words in Negative Reviews')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

nb['cells'].extend(new_cells)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Successfully appended NLP cells to odev.ipynb")
