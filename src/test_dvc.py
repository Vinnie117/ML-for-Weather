import dvc.api
import pandas as pd

with dvc.api.open(
        r'data_dvc/processed/train.csv',
        repo='https://github.com/Vinnie117/ML-for-Weather'
        ) as fd:
    train = pd.read_csv(fd, delimiter=',', header=0)


print(train.head)