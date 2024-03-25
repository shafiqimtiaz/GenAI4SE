# How to run the script

``` python finetune_gemma.py --data_dir [data directory] --max_epochs [num of epochs] --incl_ocomment [True/False] --incl_inst [True/False] --batch_size [batch size] --max_new_tokens [num of new tokens]```

```
Args:
    --data_dir: path to dataset directory.
    --max_epochs: number of epochs to train the model
    --incl_ocomment: True if you want to include this in input else False (default = False)
    --incl_inst: True if you want to include instruction in input else False (default = False)
    --batch_size: number of instances in a single batch (default = 8)
    --max_new_tokens: maximum number of new tokens to generate (default = 128)
```

1. If incl_ocomment = False and incl_inst = False then,
    input: new_code
    target: new_comment

2. If incl_ocomment = True and incl_inst = False then,
    input: old_comment, new_code
    target: new_comment

3. If incl_ocomment = True and incl_inst = True then,
    input: instruction, old_comment, new_code
    target: new_comment