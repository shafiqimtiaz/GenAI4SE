# How to run the script

``` python finetune_gemma.py [args]```

```
Args:
    --data_dir: path to dataset directory.
    --experiment: experiment number you want to execute
                    1: code >> comment
                    2: old comment, new code >> new comment
                    3: old comment, old code, new code >> new comment
                    4: old comment, old code, git-diff >> new comment
    --max_epochs: number of epochs to train the model
    --batch_size: number of instances in a single batch (default = 8)
    --max_new_tokens: maximum number of new tokens to generate (default = 128)
```