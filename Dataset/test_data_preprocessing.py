import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the JSON file into a DataFrame
df_test = pd.read_json(r"D:\Me\concordia\Notes\GenerativeAI\Project\Implementation\dataset\dataset\test.jsonl", lines=True)

# List to store instances to be removed. Not required but kept to evaluate manually..
same_comment_instances_to_remove = []

# Iterate through DataFrame rows.
for index, row in df_test.iterrows():
    src_text = row['src_javadoc']
    dst_text = row['dst_javadoc']
    
    # Check for same src and dst comment
    if src_text == dst_text:
        same_comment_instances_to_remove.append(index)

df_filtered = df_test.drop(same_comment_instances_to_remove)




# logic to remove comments which are less than 7 words in source and destination comment..

def is_comment_length_sufficient(comment):
    tokens = comment.split() 
    return len(tokens) >= 7 
instances_less_than_7_words = []

for index, row in df_test.iterrows():
    src_text = row['src_javadoc']
    dst_text = row['dst_javadoc']
    
    # Check if either src_javadoc or dst_javadoc has less than 7 tokens
    if not (is_comment_length_sufficient(src_text) and is_comment_length_sufficient(dst_text)):
        instances_less_than_7_words.append(index)

df_filtered = df_test.drop(instances_less_than_7_words)


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
# print(df_filtered.iloc[10])