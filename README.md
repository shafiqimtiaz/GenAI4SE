# Updating Method-Level Comments using Generative AI

This study explores the feasibility of using GitDiff to automatically update method-level comments with GenAI. For this, we leverage the information contained in GitDiff, i.e., a patch representing the changes between two states of the file, to help infer the modifications made to the code for updating the method-level comments. For our study as depicted by below figure, we evaluate the following two GenAI architectures:
1. CodeT5 (small): A code-specific text-to-text transformer from the T5 family.
2. Gemma 2B IT: A decoder-only LLM released by Google.

![Illustration of Methodology](https://github.com/shafiqimtiaz/GenAI4SE/blob/main/images/methodology.png "Illustration of Methodology")

To study the effectiveness of the Git-Diff parameter when it comes to improving the quality of updated comments, this study conducts four different experiments shown in the figure below: 
![Illustration of Experiments](https://github.com/shafiqimtiaz/GenAI4SE/blob/main/images/experiments.png "Illustration of Experiments")

We obtained the following average METEOR score for CodeT5-small and Gemma 2B IT:

|Experiment                        |CodeT5-small|Gemma 2B IT|
|----------------------------------|------------|-----------|
| New Code                         |0.1127      |-          |
| Old Comment + New Code (Baseline)|0.2384      |0.05095    |
| Old Code + Old Comment + New Code|0.6942      |-          | 
| Old Code + Old Comment + Git-Diff|0.7608      |0.05794    |

> _Note_: Due to computational constraints, we could not evaluate the performance of Gemma on experiments with "New Code" only and "Old Code + Old Comment + New Code". Additionally, we have not used any instruction while fine-tuning and evaluating Gemma for this task.

### Dataset
The dataset used for our study can be found [here](https://osf.io/h7s52?view_only=4a72d61422514b5ead6b263eaf512d89). The dataset is already split into train-valid-test. Execute the respective scripts under the Dataset directory to apply the necessary preprocessing and compute GitDiff from `<OldCode>` and `<NewCode>`.

### Executing CodeT5 or Gemma 2B IT
1. CodeT5:
   1. The fine-tuning script can be found under the "Training Script T5" directory
   2. To execute the script on Cluster, run the `T5.sh` file.
   3. To execute the script locally, use the below command:
        ```
        python TrainingScript.py [OPTIONS]
        ```

2. Gemma 2B IT:
    1. The fine-tuning script can be found under the "Training-Script-Gemma" directory
    2. To execute the script on the Canada Compute's Narval Cluster, run the `launch_training_accelarate.sh ` file.
    3. To execute the script locally, use the below command:
        ```
        python finetuning_gemma_for_cc.py [OPTIONS]

        OPTIONS:
         data_dir: path to the data directory
         experiment: experiment number. Default = 4
         max_epochs: maximum number of epochs to train Gemma. Default = 1
         batch_size: batch size to use for training, validation, and testing. Default = 4
         max_new_tokens: maximum number of tokens to be generated by Gemma. Default = 128  
        ```


