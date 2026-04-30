import json

notebook_path = '/Users/baris/Projects/halilinodevi/odev.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

executive_summary = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Executive Summary\n",
        "\n",
        "## Project Objective\n",
        "The objective of this project is to develop a comprehensive customer segmentation and behavioral analysis pipeline. By leveraging transactional data and customer reviews, we aim to identify distinct customer cohorts, understand their purchasing habits, and analyze their sentiment towards products. This enables highly targeted marketing strategies, improved customer retention, and data-driven product decisions.\n",
        "\n",
        "## Data Sources Used\n",
        "- **Transactional Data**: The *Online Retail Dataset* (via Kaggle), containing over 500,000 retail transactions, used for behavioral modeling.\n",
        "- **Customer Reviews Data**: A subset of the *Datafiniti Amazon Consumer Reviews* dataset, used for Natural Language Processing (NLP) and sentiment analysis.\n",
        "\n",
        "## Methods Applied\n",
        "- **RFM Analysis**: Calculated Recency, Frequency, and Monetary metrics to quantify customer value.\n",
        "- **K-Means Clustering**: Applied unsupervised machine learning to group customers into behavioral segments, validated via the Elbow Method and Silhouette Score.\n",
        "- **NLP & TF-IDF**: Processed unstructured text data using NLTK (tokenization, lemmatization) and TF-IDF vectorization to extract key themes.\n",
        "- **Sentiment Analysis**: Trained a Logistic Regression classifier to predict review sentiment based on textual features.\n",
        "\n",
        "## Key Findings & Business Impact\n",
        "- **Segmented Customer Base**: Successfully identified 4 distinct customer segments: VIPs, Loyal Customers, New/At-Risk, and Lost. \n",
        "- **Revenue Concentration**: The VIP and Loyal segments drive a disproportionate amount of revenue despite being a smaller fraction of the customer base, highlighting the need for specialized retention programs.\n",
        "- **Sentiment Drivers**: The NLP pipeline successfully identified specific keywords associated with positive and negative customer experiences, providing immediate feedback loops for product and service improvement.\n",
        "- **Actionability**: The model outputs can be directly integrated into CRM platforms to trigger automated win-back campaigns for 'Lost' customers and exclusive rewards for 'VIPs'.\n",
        "\n",
        "---"
    ]
}

insights_and_conclusion = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Comprehensive Business Insights\n",
            "\n",
            "Having completed the technical pipeline, we translate our statistical findings into actionable business intelligence."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8.1 RFM Insights\n",
            "- **The Power of RFM**: By scoring customers from 1-5 across Recency, Frequency, and Monetary value, we created an objective ranking system. A customer with a score of '555' is mathematically proven to be our best asset.\n",
            "- **Segment Profiling**:\n",
            "  - **VIPs (Score ~15)**: High spenders who buy constantly and bought recently. They require white-glove service, early access to new products, and personalized communication.\n",
            "  - **Loyal (Score 10-14)**: The backbone of the business. They need standard loyalty rewards to push them into the VIP category.\n",
            "  - **At-Risk (Score 5-9)**: Customers whose recency is slipping. They need targeted discounts or engagement emails to prevent churn.\n",
            "  - **Lost (Score ~3)**: Dormant customers with low historic value. Broad win-back campaigns can be applied, but ROI here is lowest."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8.2 Clustering Insights\n",
            "- **Real-World Representation**: The K-Means algorithm naturally discovered the same segments that traditional RFM logic dictates, proving that these behavioral differences are statistically inherent in the data, not just theoretical constructs.\n",
            "- **Behavioral Divergence**: The PCA visualization explicitly shows that VIPs operate in a completely different feature space than Lost customers. This means marketing collateral sent to a VIP will fundamentally fail if sent to a Lost customer; their motivations and engagement levels are polar opposites."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8.3 NLP Insights\n",
            "- **Sentiment Trends**: The Logistic Regression model demonstrates that text data is highly predictive of customer satisfaction. By monitoring the output of this model on live incoming reviews, the business can track real-time sentiment without waiting for sales metrics to drop.\n",
            "- **Thematic Keywords**: The TF-IDF and Bag-of-Words analysis isolated exact phrases driving satisfaction (e.g., specific product features) and dissatisfaction (e.g., shipping delays or defects). This is actionable intelligence for the product and operations teams."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Final Conclusion\n",
            "\n",
            "### What Was Achieved\n",
            "We successfully engineered an end-to-end data science pipeline that transforms raw transactional and textual data into a structured segmentation engine. We moved from messy, unstructured data to clean RFM metrics, applied robust unsupervised learning (K-Means), validated the clusters statistically, and paired behavioral data with NLP-driven sentiment analysis.\n",
            "\n",
            "### Limitations of the Analysis\n",
            "- **Static Snapshot**: RFM and K-Means were applied to a static historical snapshot. Customer behavior is fluid, and segments will drift over time.\n",
            "- **Contextual Nuance in NLP**: Our NLP approach utilized TF-IDF and Logistic Regression. While effective, it relies on word frequencies and struggles with complex contextual nuances like sarcasm or double negatives.\n",
            "- **Lack of Demographic Data**: Our segmentation relies purely on behavioral metrics. Adding demographic data (age, location, income) could significantly enrich the cluster profiles.\n",
            "\n",
            "### Possible Improvements\n",
            "- **Real-Time Segmentation**: Implement an automated pipeline that updates customer segments weekly, allowing for dynamic email marketing triggers.\n",
            "- **Advanced ML Models**: Transition from K-Means to density-based algorithms like DBSCAN if cluster shapes are non-spherical, or use predictive models (XGBoost) to forecast Customer Lifetime Value (CLV).\n",
            "- **Deep Learning for NLP**: Replace the TF-IDF approach with a pre-trained Transformer model (like BERT or RoBERTa) to capture deep semantic meaning and significantly improve sentiment classification accuracy."
        ]
    }
]

# Insert executive summary at index 1 (after the title, or at index 0)
nb['cells'].insert(0, executive_summary)

# Append insights and conclusion at the end
nb['cells'].extend(insights_and_conclusion)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook finalization complete.")
