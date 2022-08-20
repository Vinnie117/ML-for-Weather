import dvc.api
import pandas as pd


# fd = 'A:\\Projects\\ML-for-Weather\\data_dvc\\processed\\train.csv'
with dvc.api.open(
        r'data_dvc/processed/train.csv'
        ) as fd:
    print(fd)
    train = pd.read_csv(fd, delimiter=',', header=0)

# with dvc.api.open(
#         r'data_dvc/processed/train.csv',
#         repo='https://github.com/Vinnie117/ML-for-Weather'
#         ) as fd:
#     train = pd.read_csv(fd, delimiter=',', header=0)

print(train.head)