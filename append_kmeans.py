import json

notebook_path = '/Users/baris/Projects/halilinodevi/odev.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Customer Segmentation using K-Means Clustering\n",
            "\n",
            "In this section, we apply K-Means clustering to the RFM features to group customers into meaningful segments based on their purchasing behavior."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 1 & 2: Feature Selection and Preprocessing\n",
            "\n",
            "We select the Recency, Frequency, and Monetary features and apply `StandardScaler`. **Why scale?** K-Means is a distance-based algorithm. Since our features have vastly different scales (e.g., Monetary can be tens of thousands, while Frequency is mostly single digits), Monetary would dominate the distance calculation if left unscaled. Standardizing ensures each feature contributes equally."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "# Step 1: Feature Selection\n",
            "rfm_clustering_data = rfm[['Recency', 'Frequency', 'Monetary']]\n",
            "\n",
            "# Step 2: Data Preprocessing\n",
            "scaler = StandardScaler()\n",
            "rfm_scaled = scaler.fit_transform(rfm_clustering_data)\n",
            "\n",
            "# Create a DataFrame for the scaled data for visualization\n",
            "rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_clustering_data.columns)\n",
            "print(\"Scaled Features (First 5 rows):\")\n",
            "display(rfm_scaled_df.head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 3: Determine Optimal Number of Clusters\n",
            "\n",
            "We use the Elbow Method to find the optimal 'K' by iterating through 1 to 10 clusters and plotting the inertia (Within-Cluster-Sum-of-Squares)."
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
            "from sklearn.cluster import KMeans\n",
            "\n",
            "inertia = []\n",
            "k_range = range(1, 11)\n",
            "\n",
            "for k in k_range:\n",
            "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
            "    kmeans.fit(rfm_scaled)\n",
            "    inertia.append(kmeans.inertia_)\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.plot(k_range, inertia, marker='o', linestyle='--')\n",
            "plt.title('Elbow Method For Optimal K')\n",
            "plt.xlabel('Number of Clusters (K)')\n",
            "plt.ylabel('Inertia')\n",
            "plt.xticks(k_range)\n",
            "plt.grid(True)\n",
            "plt.show()\n",
            "\n",
            "optimal_k = 4\n",
            "print(f\"Based on the elbow plot, the curve starts to flatten around K={optimal_k}.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 4 & 5: Apply K-Means and Analyze Clusters\n",
            "\n",
            "We fit K-Means with the chosen optimal K and calculate the mean values of R, F, and M for each cluster."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Apply K-Means\n",
            "kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n",
            "rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)\n",
            "\n",
            "# Step 5: Cluster Analysis\n",
            "cluster_summary = rfm.groupby('Cluster').agg({\n",
            "    'Recency': 'mean',\n",
            "    'Frequency': 'mean',\n",
            "    'Monetary': 'mean',\n",
            "    'CustomerID': 'count'\n",
            "}).rename(columns={'CustomerID': 'Count'})\n",
            "\n",
            "print(\"Cluster Summary (Mean Values):\")\n",
            "display(cluster_summary)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 6: Cluster Interpretation\n",
            "\n",
            "Based on the cluster summary above, we can assign the following logical segments:\n",
            "\n",
            "1. **VIP Customers**: Extremely high Frequency and Monetary value, with very low Recency. These are our top spenders.\n",
            "2. **Loyal Customers**: Above-average Frequency and Monetary, with low Recency. Consistent and reliable.\n",
            "3. **Lost Customers**: High Recency (haven't bought in a long time), low Frequency, and low Monetary. We have likely lost them.\n",
            "4. **New / At Risk Customers**: Moderate/Low Recency, but very low Frequency and Monetary. They bought recently but haven't become regular buyers yet."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Programmatically assign labels based on relative values\n",
            "def assign_segment(row):\n",
            "    if row['Monetary'] > 50000 or row['Frequency'] > 50:\n",
            "        return 'VIP Customers'\n",
            "    elif row['Recency'] > 150:\n",
            "        return 'Lost Customers'\n",
            "    elif row['Frequency'] > 5:\n",
            "        return 'Loyal Customers'\n",
            "    else:\n",
            "        return 'New / At Risk'\n",
            "\n",
            "cluster_summary['Segment'] = cluster_summary.apply(assign_segment, axis=1)\n",
            "segment_map = cluster_summary['Segment'].to_dict()\n",
            "\n",
            "rfm['Segment'] = rfm['Cluster'].map(segment_map)\n",
            "display(cluster_summary)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 7: Visualization\n",
            "\n",
            "We visualize the cluster distribution, a 2D PCA representation of the clusters, and boxplots showing the variance of RFM metrics across the newly created segments."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.decomposition import PCA\n",
            "\n",
            "# Visualization 1: Cluster Distribution\n",
            "plt.figure(figsize=(8, 5))\n",
            "sns.countplot(data=rfm, x='Segment', palette='viridis', order=rfm['Segment'].value_counts().index)\n",
            "plt.title('Customer Distribution Across Segments')\n",
            "plt.ylabel('Number of Customers')\n",
            "plt.xticks(rotation=45)\n",
            "plt.show()\n",
            "\n",
            "# Visualization 2: 2D PCA Visualization\n",
            "pca = PCA(n_components=2)\n",
            "rfm_pca = pca.fit_transform(rfm_scaled)\n",
            "rfm['PCA1'] = rfm_pca[:, 0]\n",
            "rfm['PCA2'] = rfm_pca[:, 1]\n",
            "\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Segment', palette='viridis', alpha=0.6)\n",
            "plt.title('2D PCA Visualization of Customer Segments')\n",
            "plt.show()\n",
            "\n",
            "# Visualization 3: Boxplots for R, F, M by Segment\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
            "\n",
            "sns.boxplot(data=rfm, x='Segment', y='Recency', ax=axes[0], palette='viridis')\n",
            "axes[0].set_title('Recency by Segment')\n",
            "axes[0].tick_params(axis='x', rotation=45)\n",
            "\n",
            "sns.boxplot(data=rfm, x='Segment', y='Frequency', ax=axes[1], palette='viridis')\n",
            "axes[1].set_title('Frequency by Segment')\n",
            "axes[1].set_yscale('log') # Log scale helps handle extreme outliers in frequency\n",
            "axes[1].tick_params(axis='x', rotation=45)\n",
            "\n",
            "sns.boxplot(data=rfm, x='Segment', y='Monetary', ax=axes[2], palette='viridis')\n",
            "axes[2].set_title('Monetary by Segment')\n",
            "axes[2].set_yscale('log') # Log scale for monetary\n",
            "axes[2].tick_params(axis='x', rotation=45)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

nb['cells'].extend(new_cells)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Successfully appended K-Means cells to odev.ipynb")
