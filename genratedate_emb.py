import random
from typing import List
import os
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler



import pandas as pd


def time2_features(
    dates,
    main_df=None,
    use_features=["year", "month", "day", "weekday", "hour", "minute"],
):
    if main_df is None:
        main_df = pd.DataFrame({})
    else:
        main_df = main_df.copy()
    years = dates.apply(lambda row: row.year)
    max_year = years.max()
    min_year = years.min()

    if "year" in use_features:
        main_df["Year"] = dates.apply(
            lambda row: (row.year - min_year) / max(1.0, (max_year - min_year))
        )

    if "month" in use_features:
        main_df["Month"] = dates.apply(
            lambda row: 2.0 * ((row.month - 1) / 11.0) - 1.0, 1
        )
    if "day" in use_features:
        main_df["Day"] = dates.apply(lambda row: 2.0 * ((row.day - 1) / 30.0) - 1.0, 1)
    if "weekday" in use_features:
        main_df["Weekday"] = dates.apply(
            lambda row: 2.0 * (row.weekday() / 6.0) - 1.0, 1
        )
    if "hour" in use_features:
        main_df["Hour"] = dates.apply(lambda row: 2.0 * ((row.hour) / 23.0) - 1.0, 1)
    if "minute" in use_features:
        main_df["Minute"] = dates.apply(
            lambda row: 2.0 * ((row.minute) / 59.0) - 1.0, 1
        )

    main_df["Datetime"] = dates
    return main_df

if __name__ == "__main__":
        time_features: List[str] = [
            "year",
            "month",
            "day",
            "weekday",
#            "hour",
#            "minute",
        ]

        raw_df = pd.read_csv("./toy-time.csv")
        raw_df_notime = pd.read_csv("./toy.csv")
        time_df = pd.to_datetime(raw_df['Datetime'], format="%Y-%m-%d",errors='coerce') #errors='coerce'
        #time_df = raw_df[0]
        df = time2_features(
            time_df, raw_df, use_features=time_features
        )
        print("raw_df",raw_df.shape)
        print("df.shape", df.shape)
        time_cols = df.columns.difference(raw_df.columns)
        print("raw_df_notime",raw_df_notime.shape)
        #df = df.columns.difference(raw_df.columns)
        print("self.time_cols", time_cols)
        print("df2", df)
        print("self.time_cols.shape", time_cols.shape)
        cx = df[time_cols]
        cx = np.array(cx)
        # cx = pd.DataFrame(cx)
        # print("cx.shape",cx.shape)  #(52560,6)  (l,f)->(l,n,f)
        x_t = torch.Tensor(cx).unsqueeze(1).repeat(1, 20, 1)
        x_t = np.array(x_t)
        x_t = torch.Tensor(x_t)
        print("x_t.shape", x_t.shape)
        print("x_t", x_t)

        raw_df_notime = np.array(raw_df_notime)
        raw_df_notime = torch.Tensor(raw_df_notime)
        x = torch.cat((raw_df_notime.unsqueeze(-1),x_t),dim = -1)
        print("x.shape",x.shape)
        print("x",x)

        np.savez("./toy_all", x)

