import pandas as pd

from .config import TitanicRunnerConfigs
from .constant import MAIN, PRODUCTION, SPLIT, TEST, TRAIN


def get_df(cfg: TitanicRunnerConfigs):
    df_main = pd.read_csv(cfg.data_path / "train.csv", nrows=cfg.load_data_row_nums)
    df_test = pd.read_csv(cfg.data_path / "test.csv", nrows=cfg.load_data_row_nums)

    df_main[SPLIT] = TRAIN
    df_test[SPLIT] = PRODUCTION

    df_main = pd.concat([df_main, df_test]).reset_index(drop=True)

    return {MAIN: df_main}
