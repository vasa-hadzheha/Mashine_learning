import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
med = pd.read_csv("Med.csv")

# Extract relevant columns for one-hot encoding
columns_to_extract = ['Категорії англійською', 'Гість_']
one_hot = med[columns_to_extract]

# Perform one-hot encoding
one_hot_encoded = one_hot.groupby(['Гість_', 'Категорії англійською']).size().unstack(fill_value=0)
one_hot_encoded_df=pd.DataFrame(one_hot_encoded)
column_names = one_hot_encoded.columns
print(column_names)
#one_hot_encoded_df = one_hot_encoded_df.drop(columns=['Категорії англійською'])
print(one_hot_encoded_df.head())
# Standardize the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(one_hot_encoded)
# scaled_df = pd.DataFrame(scaled_data)
# print(scaled_df)
# BIRCH clustering
birch = Birch(threshold=0.5, n_clusters=3)
birch.fit(one_hot_encoded)

# Get the labels
cluster_labels = birch.labels_

# Add the labels to the original DataFrame
one_hot_encoded['cluster'] = cluster_labels

# t-SNE Visualization
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(one_hot_encoded)

# Create a DataFrame for t-SNE results
tsne_df = pd.DataFrame(tsne_result, columns=['tsne1', 'tsne2'])
tsne_df['cluster'] = cluster_labels

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tsne1', y='tsne2', hue='cluster', data=tsne_df, palette='viridis', legend='full')
plt.title('t-SNE Visualization of Birch Clustering')
plt.show()
