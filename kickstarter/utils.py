import pandas as pd


def summarize_to_others(df, col_name, thread):
    origin_col_dict = df[col_name].value_counts().to_dict()
    new_df = df[col_name].apply(lambda x: x if origin_col_dict[x] > thread else 'others')
    return new_df