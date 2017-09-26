# Jupyter Notebooks for Deep Learning Course

Forked from the repository containing jupyter notebooks for the book [Deep Learning with Python (Manning Publications)](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff)

## Prerequisites

- Install Python 3.6
- Install requirements with ```pip install -r requirements.txt```
- Download Kaggle Cats and Dogs dataset from [this link](https://drive.google.com/file/d/0B_ebsCRJm2BfZU9Ib0FVaHJHOEU/view?usp=sharing)
- Extract it under the ```data/``` directory. After extraction you should have the following structure:

```shell
data/
data/kaggle_original_data/
data/kaggle_original_data/cat.0.jpg
data/kaggle_original_data/cat.1.jpg
...
```

## GCloud Instructions

- Machines are already prepared with requirements, code & data.
- Access machine with: ```ssh -i key_file dlcv@IP```
- Navigate to ```dl-mediapro```
- Start jupyter notebook with: ```jupyter notebook```. Default port for Jupyter is 8123.
- Open tunnel to edit notebook locally: ```ssh -i key_file -L 8899:localhost:8123 dlcv@IP```
- Navigate to ```localhost:8899``` from your local browser and start editing.
- Similarly, if you want to use tensorboard, run: ```tensorboard --logdir=whatever  --port=8008```. And then open another tunnel to forward this port.

Note: IPs, ssh keys and passwords will be provided in class.