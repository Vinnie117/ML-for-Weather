def feature_engineering(cfg: data_config):

    pipe = Pipeline([
        ("split", Split(test_size= cfg.model.split, shuffle = cfg.model.shuffle)), # -> sklearn.model_selection.TimeSeriesSplit
        ("times", Times()),
        ("lags", InsertLags(vars=cfg.transform.vars, diff=cfg.diff.lags)),
        ('velocity', Velocity(vars=cfg.transform.vars, diff=cfg.diff.velo)),   
        ('lagged_velocity', InsertLags(vars=cfg.transform.lags_velo, diff=cfg.diff.lagged_velo)),     # lagged difference = differenced lag
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.acc)),                   # diff of rows (s) between 2 subsequent velos
        ('lagged_acceleration', InsertLags(vars=cfg.transform.lags_acc, diff=cfg.diff.lagged_acc)),   
        ('cleanup', Prepare(target = cfg.model.target, vars=cfg.model.predictors))
        ])

    return pipe


    class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically insert lags (compute new features in 'X', add to master 'data')
    """
    def __init__(self, vars, diff):
        self.diff = diff
        self.vars = vars

    def fit(self, X):
        return self

    def transform(self, X):

        data = copy.deepcopy(X)

        # create column names
        cols = []
        for i in range(len(self.diff)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_lag_' + str(self.diff[i]))

        # create data (lags) for each data set k (train/test) in dict X
        for k, v in X.items():
            col_indices = [data[k].columns.get_loc(c) for c in self.vars if c in data[k]]
            dummy = []
            for i in self.diff:
                dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].shift(i)))
            X[k] = pd.concat(dummy, axis=1)
            X[k].columns = cols
 
            # combine with master data frame
            data[k] = pd.concat([data[k], X[k]], axis=1)


        return data # a dict with training and test data
