import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
import os
# Read the JSON file into a DataFrame
df_train = pd.read_json("../train.jsonl", lines=True)


# List to store instances to be removed. Not required but kept to evaluate manually..
same_comment_instances_to_remove = []

# Iterate through DataFrame rows.
for index, row in df_train.iterrows():
    src_text = row['src_javadoc']
    dst_text = row['dst_javadoc']  
    # Check for same src and dst comment..
    if src_text == dst_text:
        same_comment_instances_to_remove.append(index)

df_filtered = df_train.drop(same_comment_instances_to_remove)

# Tokenize src_javadoc and dst_javadoc for each row
df_filtered['src_javadoc_tokens'] = df_filtered['src_javadoc'].apply(lambda x: x.split())
df_filtered['dst_javadoc_tokens'] = df_filtered['dst_javadoc'].apply(lambda x: x.split())

#Reducing the training set size for gemma
df_filtered['src_javadoc'] = df_filtered['src_javadoc_tokens'].apply(lambda x : len(x) > 7)
df_filtered['dst_javadoc'] = df_filtered['dst_javadoc_tokens'].apply(lambda x : len(x) > 7)

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

# adding diff column in the dataset...

 
def generate_diff(old_code, new_code):
    old_code_lines = old_code.splitlines()
    new_code_lines = new_code.splitlines()
 
    diff = difflib.unified_diff(old_code_lines, new_code_lines)
 
    return '\n'.join(list(diff))

# Apply the generate_diff function to each pair of 'src_method' and 'dst_method' and store the result in a new column 'diff'
df_filtered['diff'] = df_filtered.apply(lambda row: generate_diff(row['src_method'], row['dst_method']), axis=1)

directory = "D:/Me/concordia/Notes/GenerativeAI/Project/Implementation/gemma/"
if not os.path.exists(directory):
    os.makedirs(directory)

# Storing filtered data to a csv file.
csv_file_path = r"D:/Me/concordia/Notes/GenerativeAI/Project/Implementation/gemma/train_preprocessed.csv"

df_filtered.to_csv(csv_file_path, index=False)
print(df_filtered.columns)
print(df_filtered.shape)
print(df_train.shape)
# print(df_filtered.iloc[10])