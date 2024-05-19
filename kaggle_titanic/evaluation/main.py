from ml_assemblr.main_components.data_pod import DataPod

from .agg_results import agg_df_evaluation, agg_shap_results
from .get_cv_results import get_df_cv_evaluation, get_shap_values


def evaluation(dp: DataPod, objective_name: str = "accuracy_valid", fast_mode: bool = False) -> DataPod:
    is_cv = "cv_idx_map" in dp.variables
    cv_count = len(dp.variables["cv_idx_map"]["cv_pred_idx_in_column_type"]) if is_cv else 1
    dfs_evaluation = []
    shap_results = []

    # get results for each cross validation =======
    for cv_idx in range(cv_count):
        # df_evaluation
        df_cv_evaluation = get_df_cv_evaluation(dp, is_cv, cv_idx)
        dfs_evaluation.append(df_cv_evaluation)

        if not fast_mode:
            # shap
            cv_shap_results = get_shap_values(dp, is_cv, cv_idx)
            shap_results.append(cv_shap_results)

    # aggregation ==================================
    dp.variables["evaluation"] = {}
    dp = agg_df_evaluation(dp, dfs_evaluation, is_cv, objective_name)

    if not fast_mode:
        # shap
        dp = agg_shap_results(dp, shap_results, is_cv, cv_count)

    return dp
