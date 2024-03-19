import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the JSON file into a DataFrame
df_test = pd.read_json(r"D:\Me\concordia\Notes\GenerativeAI\Project\Implementation\dataset\dataset\test.jsonl", lines=True)


# function to compute cosine similarity
def compute_cosine_similarity(src_text, dst_text):
   
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([src_text, dst_text])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    return cosine_sim

# Define a threshold for similarity
similarity_threshold = 0.95

# List to store instances to be removed. Not required but kept to evaluate manually..
instances_to_remove = []

# Iterate through DataFrame rows.
for index, row in df_test.iterrows():
    src_text = row['src_javadoc']
    dst_text = row['dst_javadoc']
    
    similarity = compute_cosine_similarity(src_text, dst_text)
    
    # Check if similarity is above the threshold..
    if similarity > similarity_threshold:
        instances_to_remove.append(index)

df_filtered = df_test.drop(instances_to_remove)

# Tokenize src_javadoc and dst_javadoc for each row
df_filtered['src_javadoc_tokens'] = df_filtered['src_javadoc'].apply(lambda x: x.split())
df_filtered['dst_javadoc_tokens'] = df_filtered['dst_javadoc'].apply(lambda x: x.split())

# Compute min and max token count of src_javadoc and dst_javadoc
min_src_tokens = df_filtered['src_javadoc_tokens'].apply(len).min()
max_src_tokens = df_filtered['src_javadoc_tokens'].apply(len).max()
min_dst_tokens = df_filtered['dst_javadoc_tokens'].apply(len).min()
max_dst_tokens = df_filtered['dst_javadoc_tokens'].apply(len).max()

print("Minimum token count of src_javadoc:", min_src_tokens)
print("Maximum token count of src_javadoc:", max_src_tokens)
print("Minimum token count of dst_javadoc:", min_dst_tokens)
print("Maximum token count of dst_javadoc:", max_dst_tokens)

# dropping unnecessary columns..
columns_to_drop = ['code_change_seq', 'index', 'src_desc', 'dst_desc', 'src_desc_tokens', 'dst_desc_tokens', 'desc_change_seq', 'dist', 'dst_javadoc_tokens', 'src_javadoc_tokens']
df_filtered.drop(columns=columns_to_drop, inplace=True)
# Storing filtered data to a csv file.
csv_file_path = r"D:\Me\concordia\Notes\GenerativeAI\Project\Implementation\test_preprocessed.csv"

df_filtered.to_csv(csv_file_path, index=False)
print(df_filtered.columns)
print(df_filtered.shape)
print(df_test.shape)
print(df_filtered.iloc[10])