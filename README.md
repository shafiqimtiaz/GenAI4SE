# Updating Method-Level Comments using Generative AI

This study explores the feasibility of using GitDiff to automatically update method-level comments with GenAI. For this, we leverage the information contained in GitDiff, i.e., a patch representing the changes between two states of the file, to help infer the modifications made to the code for updating the method-level comments. For our study, we evaluate the following two GenAI architectures:
1. CodeT5 (small): A code-specific text-to-text transformer from the T5 family.
2. Gemma 2B IT: A decoder-only LLM released by Google. 

To study the effectiveness of the Git-Diff parameter when it comes to improving the quality of updated comments, this study conducts four different experiments shown in the figure below:

