
def to_monthly(df, col, how="last"):
    df = df.dropna().copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby("month")[col]
    if how == "last":
        out = g.last()
    elif how == "mean":
        out = g.mean()
    elif how == "sum":
        out = g.sum()
    else:
        raise ValueError("how must be last|mean|sum")
    return out.reset_index().rename(columns={"month":"ds"})
