  File "/data/anaconda3/envs/py311/lib/python3.11/site-packages/transformers/trainer.py", line 1954, in _wrap_model
    if self.accelerator.unwrap_model(model, keep_torch_compile=False) is not model:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'
  

RuntimeError: Expected to mark a variable ready only once. This error is caused by one

gradient_checkpoint设置为False