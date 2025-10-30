import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error


if __name__ == "__main__":

    print('Preparing Data...')
    with open('../data/DatasetCleaned.csv', 'r') as f:
        data = pd.read_csv(f)


    Y_DATA = np.log(data['SalePrice'].copy().values)
    VAR_DEPENDENT = 'SalePrice'

    X_DATA = data.copy().drop(columns=['SalePrice','Unnamed: 0'])

    def PrepX(X:pd.DataFrame=X_DATA):

        X_prep = pd.DataFrame(index=X.index)
        X_types = X.dtypes.to_dict()

        new_cols = []
        for col, t in X_types.items():

            if t in ['int', 'float']: 
                new_cols.append(X[col].copy())
                continue

            if t == 'object': 
                new_col = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
                new_cols.append(new_col)
                continue

            print(f'Type \"{t}\" not in standard types!')

        if new_cols: X_prep = pd.concat([X_prep] + new_cols, axis=1)

        var_names = X_prep.columns
        X_prep = X_prep.values.reshape(-1, X_prep.shape[1])

        return X_prep, var_names


    X_DATA, VAR_NAMES = PrepX()

    print('Searching Grid...')

    ForestModel = RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=42)
    param_dict = {
        'max_depth':list(range(3,11)),
        'min_samples_leaf':[5,10,25,50,100],
        'max_features':['sqrt','log2',10]
    }
    scorer = {
        'R2':make_scorer(r2_score)
    }

    TuningGrid = GridSearchCV(
        ForestModel, param_dict,
        scoring=scorer, refit='R2', cv=KFold(5), return_train_score=True,
        verbose=3, n_jobs=-1
    )

    TuningGrid.fit(X_DATA,Y_DATA)

    with open('../models/GridSearchedRF.sav', 'wb') as f: pickle.dump(TuningGrid, f)