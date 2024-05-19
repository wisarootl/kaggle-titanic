import pandas as pd
from ml_assemblr.main_components.data_pod import DataPod

from kaggle_titanic.constant import TEST, TRAIN, VALID


def agg_df_evaluation(
    dp: DataPod, dfs_evaluation: list[pd.DataFrame], is_cv: bool, objective_name: str
) -> DataPod:
    df_evaluation = dfs_evaluation[0]
    if is_cv:
        for df_cv_evaluation in dfs_evaluation[1:]:
            df_evaluation = pd.merge(df_evaluation, df_cv_evaluation, on="metric")

        df_evaluation["value_mean"] = df_evaluation.drop(columns=["metric"]).mean(axis=1)
        df_evaluation["value_std"] = df_evaluation.drop(columns=["metric"]).std(axis=1)

    dp.variables["evaluation"]["df_evaluation"] = df_evaluation

    value_col_name = "value_mean" if is_cv else "value"

    dp.variables["evaluation"][f"objective_{objective_name}"] = df_evaluation[
        df_evaluation["metric"] == objective_name
    ][value_col_name].values[0]

    return dp


def agg_shap_results(dp: DataPod, shap_results: dict, is_cv: bool, cv_count: int) -> DataPod:
    splits = (TRAIN, VALID, TEST)
    dp.variables["evaluation"]["shap_results"] = shap_results
    dp.variables["evaluation"]["agg_shap_results"] = {}
    for split in splits:

        df_shap_importance = shap_results[0][split]["df_shap_importance"]
        if df_shap_importance is not None:
            if is_cv:
                for cv_idx in range(1, cv_count):
                    df_cv_shap_importance = shap_results[cv_idx][split]["df_shap_importance"]
                    df_shap_importance = pd.merge(
                        df_shap_importance, df_cv_shap_importance, on="feature"
                    )

                df_shap_importance["value_mean"] = df_shap_importance.drop(columns=["feature"]).mean(
                    axis=1
                )
                df_shap_importance["value_std"] = df_shap_importance.drop(columns=["feature"]).std(
                    axis=1
                )

        dp.variables["evaluation"]["agg_shap_results"][
            f"df_shap_importance_{split}"
        ] = df_shap_importance
    return dp
