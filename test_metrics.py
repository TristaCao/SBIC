"""
 Get the True Positive Rate.
 To use this, you must have a pickle filee in the same order of the examples in Bias in Bios, with either True aor False depending on if the prediction is correct.
 """
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def get_performance(predictions, threshold=0.5):
    path = "SBIC/SBIC.v2.agg.tst.csv"
    gold = pd.read_csv(path)
    labels = gold["hasBiasedImplication"].apply(lambda x: 1 if x=='0' else 0).tolist()
    print("Get performance-----------------")
    print(labels[:100])
    print(predictions[:100])
    f1 = f1_score(labels, predictions)
    return f1



def get_predictions_and_evaluate(preds):
    f1_results = get_performance(preds)
    # tped_results = get_jigsaw_fairness_functions(preds)
    # fped_results = get_jigsaw_fairness_functions(preds, fairness_func=false_positive_rate, metric_name="FPED")
    print("F1:", f1_results)
    return f1_results