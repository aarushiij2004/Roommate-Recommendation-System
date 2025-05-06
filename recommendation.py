import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tabulate import tabulate


df = pd.read_json('/content/roommate_sample.json')

def preprocess(df):
    df = df.copy()
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp', axis=1)

    le = LabelEncoder()
    for col in ['Cleanliness Preference', 'Sleep Schedule', 'Comfort with Smoking/Alcohol',
                'Sports Participation', 'Noise Level', 'Social Personality',
                'Guests', 'Sharing Preferences', 'Arts/Performance Clubs', 
                'Room Cleaning Frequency', 'Indian State']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    hobbies = ['Gaming', 'Reading', 'Music', 'Sports', 'Traveling', 'Cooking', 
               'Photography', 'Painting', 'Dancing', 'Hiking']
    if 'Hobbies' in df.columns:
        for hobby in hobbies:
            df[hobby] = df['Hobbies'].apply(lambda x: int(hobby in str(x)))
        df = df.drop('Hobbies', axis=1)

    languages = ['Hindi', 'English', 'Tamil', 'Bengali', 'Marathi', 'Gujarati', 
                 'Punjabi', 'Malayalam', 'Kannada', 'Odia', 'Telugu', 'Other']
    if 'Languages Spoken' in df.columns:
        for language in languages:
            df[language] = df['Languages Spoken'].apply(lambda x: int(language in str(x)))
        df = df.drop('Languages Spoken', axis=1)

    return df

df_processed = preprocess(df)

scaler = StandardScaler()
X = scaler.fit_transform(df_processed.drop(['Full Name', 'Email Address'], axis=1))

pca = PCA(n_components=min(20, X.shape[1]))
X_reduced = pca.fit_transform(X)


similarity_matrix = cosine_similarity(X_reduced)
names = df['Full Name'].tolist()

top_n = 3
matches = {}
for i, name_i in enumerate(names):
    scores = sorted(enumerate(similarity_matrix[i]), key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx, _ in scores if idx != i][:top_n]
    matches[name_i] = [(names[j], similarity_matrix[i][j]) for j in top_idxs]

results = []
for user, recs in matches.items():
    row = {
        'User': user,
        'Top 1 Match': f"{recs[0][0]} ({recs[0][1]:.2f})",
        'Top 2 Match': f"{recs[1][0]} ({recs[1][1]:.2f})",
        'Top 3 Match': f"{recs[2][0]} ({recs[2][1]:.2f})",
    }
    results.append(row)

results_df = pd.DataFrame(results)
print("\nTop 3 Matches per Person:\n")
print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))


G = nx.Graph()
G.add_nodes_from(names)
for person, recs in matches.items():
    for match_name, score in recs:
        if not G.has_edge(person, match_name):
            G.add_edge(person, match_name, weight=score)

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
weights = [d['weight'] * 5 for (_, _, d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
plt.title("Roommate Matching Network Graph (Top 3 Similarities)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_reduced)
df['Cluster'] = labels
print("\nCluster Assignments:\n")
print(tabulate(df[['Full Name', 'Cluster']], headers='keys', tablefmt='pretty', showindex=False))


def get_common_features(i, j):
    feature_names = df_processed.drop(['Full Name', 'Email Address'], axis=1).columns
    diff = np.abs(X[i] - X[j])
    return [f for f, d in zip(feature_names, diff) if d < 0.5]

print("\nFeature Similarities Example:")
for i, (match_name, _) in enumerate(matches[names[0]]):
    j = names.index(match_name)
    common = get_common_features(0, j)
    print(f"{names[0]} & {match_name} common features: {common}")


avg_sim = similarity_matrix.mean(axis=1)
names_with_avg = list(zip(names, avg_sim))
names_with_avg.sort(key=lambda x: x[1])
cutoff_index = max(1, int(0.1 * len(names)))  
cutoff_score = names_with_avg[cutoff_index][1]

outliers = [name for name, score in names_with_avg if score <= cutoff_score]

if outliers:
    print(f"\nDetected Outliers (Bottom 10% Similarity Scores ≤ {cutoff_score:.2f}):")
    for name in outliers:
        print(f"- {name}")
else:
    print("\nNo significant outliers detected.")
