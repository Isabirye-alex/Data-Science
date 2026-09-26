from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from zlib import crc32
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path('datasets').mkdir(exist_ok=True, parents=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_datastet:
            housing_datastet.extractall(path='datasets')
    return pd.read_csv(Path('datasets/housing/housing.csv'))
housing = load_housing_data()
# print(housing.head(5))
# housing.info()
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

def shuffle_and_split_data(data: pd.DataFrame, test_ratio: float):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#train_set, test_set = shuffle_and_split_data(housing, 0.2)

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()  # adds an `index` column
housing_with_id['id'] = housing_with_id['longitude'] * 1000 + housing_with_id['latitude']
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
# print(len(train_set))
# print(train_set.head(5))
# print('----------------------------')
# print(len(test_set))
# print(test_set.head(5))

train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5,3.0,4.5,6., np.inf], labels=[1,2,3,4,5] )
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
# plt.show()

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

strat_splits = []

for train_index, test_index in splitter.split(housing, housing['income_cat']):
    strat_train_n = housing.iloc[train_index]
    strat_test_n = housing.iloc[test_index]
    strat_splits.append([strat_train_n, strat_test_n])

    strat_train_set, strat_test_set = strat_splits[0]
print(len(strat_train_set))
print('.............................')
print(len(strat_test_set))