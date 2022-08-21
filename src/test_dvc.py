import dvc.api
import pandas as pd


# fd = 'A:\\Projects\\ML-for-Weather\\data_dvc\\processed\\train.csv'
with dvc.api.open(
        r'data_dvc/processed/train.csv'
        ) as data:
    print(data)
    train = pd.read_csv(data, delimiter=',', header=0)

# with dvc.api.open(
#         r'data_dvc/processed/train.csv',
#         repo='https://github.com/Vinnie117/ML-for-Weather'
#         ) as data:
#     train = pd.read_csv(data, delimiter=',', header=0)

print(train.head)