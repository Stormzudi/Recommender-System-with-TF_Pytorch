from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import xgboost as xgb
import pandas as pd
from matplotlib import pyplot as plt


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def get_xgb_feat_importances(clf, train_features):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.get_booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()

    feat_importances = []

    for feat, value in zip(fscore.keys(), train_features):
        feat_importances.append({'Feature': value, 'Importance': fscore[feat]})

    # for ft, score in fscore.items():
    #     feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    return dict(zip(fscore.keys(), train_features)), feat_importances


def plotDataTrend(data, path):
    val = data.sort_values(
        by=['year', 'month', 'day'], ascending=True).reset_index(drop=True)

    val["date"] = val.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}-{int(x['day'])}", axis=1)
    val["date"] = val["date"].apply(lambda x: pd.to_datetime(x))

    if val.unit.values[0] in [497, 81, 9, 285, 554, 315]:
        plt.figure(figsize=(15, 6))
        plt.plot(val.date, val['qty'], 'o', label='raw_data')
        plt.plot(val.date, val['pre_qty'], 'ro', label="pre_data")
        plt.legend()
        plt.savefig('{}/output/xgb/unit_{}.jpg'.format(path, val.unit.values[0]))
        plt.title(str(val.unit.values[0]))
        plt.show()