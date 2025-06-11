Python 3.8.3 (tags/v3.8.3:6f8c832, May 13 2020, 22:37:02) [MSC v.1924 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import vtorch
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    import vtorch
ModuleNotFoundError: No module named 'vtorch'
>>> import torch
>>> x = torch.tensor(1.0)
>>> y = torch.tensor(2.0)
>>> w = torch.tensor(1)
>>> del w
>>> w = torch.tensor(1.0,requires_grad = True)
>>> 