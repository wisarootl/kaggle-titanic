from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.model.base_model import get_model_index, get_model_transformer
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from kaggle_titanic.constant import TEST, TRAIN, VALID
from ml_assemblr.transformer.model.xgb_model import XGBModel


def calculate_evaluation_metric(
    df: pd.DataFrame,
    label_col_name: str,
    pred_col_name: str,
    metric: str,
    thresholds: Optional[float] = None,
):
    if thresholds:
        classified_pred = df[pred_col_name] >= thresholds

    # classification
    if metric == "auroc":
        score = roc_auc_score(df[label_col_name], df[pred_col_name])

    # classification with threshold
    elif metric == "accuracy":
        score = (df[label_col_name] == classified_pred).mean()

    # regression
    elif metric == "rmse":
        score = root_mean_squared_error(df[label_col_name], df[pred_col_name])

    else:
        score = None
    return score


def get_df_cv_evaluation(dp: DataPod, is_cv: bool, cv_idx: Optional[int] = None):
    label_col_name = dp.main_column_type.labels[0]

    if is_cv:
        pred_col_idx_in_col_type = dp.variables["cv_idx_map"]["cv_pred_idx_in_column_type"][cv_idx]
        pred_col_name = dp.main_column_type.predictions[pred_col_idx_in_col_type]
    else:
        pred_col_name = dp.main_column_type.predictions[0]

    metrics = ("accuracy",)
    splits = (TRAIN, VALID, TEST)

    cv_evaluation = []

    for metric in metrics:
        for split in splits:
            if is_cv:
                split_idx_in_column_type = dp.variables["cv_idx_map"]["cv_split_idx_in_column_type"][
                    cv_idx
                ]
            else:
                split_idx_in_column_type = 0

            relevant_df = dp.slice_df(
                split=split,
                columns=[pred_col_name, label_col_name],
                table_name=dp.main_df_name,
                split_idx_in_column_type=split_idx_in_column_type,
            )

            if relevant_df.shape[0] > 0:
                metric_value = calculate_evaluation_metric(
                    relevant_df,
                    label_col_name,
                    pred_col_name,
                    metric,
                    thresholds=dp.variables["model_threshold"],
                )
            else:
                metric_value = None

            value_col_name = f"value_cv_{cv_idx}" if is_cv else "value"
            cv_evaluation.append({"metric": f"{metric}_{split}", value_col_name: metric_value})

    df_cv_evaluation = pd.DataFrame(cv_evaluation)
    return df_cv_evaluation


def get_shap_values(dp: DataPod, is_cv: bool, cv_idx: int = 1):
    splits = (TRAIN, VALID, TEST)
    shap_results = {}
    for split in splits:
        if is_cv:
            split_idx_in_column_type = dp.variables["cv_idx_map"]["cv_split_idx_in_column_type"][cv_idx]
        else:
            split_idx_in_column_type = 0

        relevant_df_features = dp.slice_df(
            split=split,
            columns="features",
            table_name=dp.main_df_name,
            split_idx_in_column_type=split_idx_in_column_type,
        )

        if relevant_df_features.shape[0] > 0:

            model_index = get_model_index(dp, order=cv_idx)
            model_transformer: XGBModel = get_model_transformer(dp, model_index)
            relevant_df_features = model_transformer._correct_type_for_categorical_columns(
                relevant_df_features, dp.main_column_type.categorical_features
            )

            dfeatures = xgb.DMatrix(
                data=relevant_df_features, enable_categorical=model_transformer.enable_categorical
            )

            booster = model_transformer.model
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(dfeatures)

            shap_importance = np.mean(abs(shap_values), axis=0)
            feature_names = dp.main_column_type.features
            shap_importance = dict(zip(feature_names, shap_importance, strict=True))
            value_col_name = f"value_cv_{cv_idx}" if is_cv else "value"
            df_shap_importance = pd.DataFrame(
                list(shap_importance.items()), columns=["feature", value_col_name]
            )

            df_shap_importance[value_col_name] = (
                df_shap_importance[value_col_name] / df_shap_importance[value_col_name].sum()
            ) * 100

        else:
            shap_values = None
            df_shap_importance = None

        shap_results[split] = {"shape_value": shap_values, "df_shap_importance": df_shap_importance}

    return shap_results
