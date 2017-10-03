# Jupyter Notebooks for Deep Learning Course

Forked from the repository containing jupyter notebooks for the book [Deep Learning with Python (Manning Publications)](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff)

Created in October 2017 by [Víctor Campos](https://imatge.upc.edu/web/people/victor-campos) and [Amaia Salvador](https://imatge.upc.edu/web/people/amaia-salvador), PhD candidates at Universitat Politècnica de Catalunya.

## Prerequisites

- Install Python 3.6
- Install requirements with ```pip install -r requirements.txt```
- Install PyCairo:

```shell
git clone https://github.com/pygobject/pycairo.git
cd pycairo
python setup.py install
```

## Data 

### Kaggle Cats and Dogs

- Download the preprocessed Kaggle Cats and Dogs dataset from [this link](https://mega.nz/#!l0VjyQaT!AMlPjhY36BhTBQ21bLbkRkVuoM6pagv0vg-LKFuzR-8) (86MB)
- Extract it under the ```data/``` directory. After extraction you should have the following structure:

```shell
data/
data/cats_and_dogs_small/train
data/cats_and_dogs_small/validation
data/cats_and_dogs_small/test
```

### UCF-101 (Coming soon)
- Download pre-extracted features [here]() (16GB).
- Extract them under ```data```:
```shell
data/
data/ucf101/train...
```

### Models

- The pretrained models that are required to run all the notebooks with no code changes required can be downloaded from [this](https://mega.nz/#!ts82VR4T!zJGEyFE_lW8QRDAioJ4UQV7d5Slv6gq0D0CXjpzpmI8) link (792MB). Download, extract, and place under ```data```.

```shell
data/
data/models/
data/models/model_name.pkl
...
```

## Contact

For any questions or suggestions use the issues section or drop us an e-mail at victor.campos@bsc.es or amaia.salvador@upc.edu