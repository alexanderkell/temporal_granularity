g = sns.catplot(x="num_days", y="value", hue="method", col="metric",
                    data=results_dataframe, kind="box", sharey=False)