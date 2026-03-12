from utils.constant import COLUMNS, SPLIT_DATASET
import pandas
import plotly.figure_factory as ff
import sys

def plot_dataset(df: pandas.DataFrame):
	
	df_mean: pandas.DataFrame = df.drop(columns=[c for c in df.columns if "std" in c or "worst" in c])
	df_std: pandas.DataFrame = df.drop(columns=[c for c in df.columns if "mean" in c or "worst" in c])
	df_worst: pandas.DataFrame = df.drop(columns=[c for c in df.columns if "mean" in c or "std" in c])

	fig = ff.create_scatterplotmatrix(
        df_mean,
        index="Diagnosis",
        diag="histogram",
        height=1536,
        width=2048,
        title="Relation between Mean Features",
        colormap_type="cat"
    )
	fig.update_xaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	fig.update_yaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	fig.write_html("plots/pair_plot_mean.html", auto_open=True)

	fig = ff.create_scatterplotmatrix(
        df_std,
        index="Diagnosis",
        diag="histogram",
        height=1536,
        width=2048,
        title="Relation between STD Features",
        colormap_type="cat"
    )
	fig.update_xaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	fig.update_yaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	fig.write_html("plots/pair_plot_std.html", auto_open=True)

	fig = ff.create_scatterplotmatrix(
        df_worst,
        index="Diagnosis",
        diag="histogram",
        height=1536,
        width=2048,
        title="Relation between Worst Features",
        colormap_type="cat"
    )
	fig.update_xaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	fig.update_yaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	fig.write_html("plots/pair_plot_worst.html", auto_open=True)

	# fig = ff.create_scatterplotmatrix(
    #     df,
    #     index="Diagnosis",
    #     diag="histogram",
    #     height=1536,
    #     width=2048,
    #     title="Relation between Features",
    #     colormap_type="cat"
    # )
	# fig.update_xaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	# fig.update_yaxes(showticklabels=False, showgrid=False, ticks="", zeroline=False)
	# fig.write_html("plots/pair_plot.html", auto_open=True)


def	main():
	if len(sys.argv) != 2:
		print("Error: python process_dataset.py *dataset*")
		return 1
	ds_path: str = sys.argv[1]
	try:
		df: pandas.DataFrame = pandas.read_csv(ds_path)
	except Exception as e:
		print(e)
		return 
	try:
		f_test = open("datasets/test_dataset.csv", "w")
		f_train = open("datasets/train_dataset.csv", "w")
	except:
		print("Error log: can not create dataset files")
		return 1
	
	df.columns = COLUMNS
	for x in df.columns[2:]:
		df.drop(df[df[x] == 0.0].index, inplace=True)

	plot_dataset(df.drop(columns="ID"))

	slice:int = int(len(df) / 100) * SPLIT_DATASET
	df_train = df.iloc[:slice, :]
	df_test = df.iloc[slice+1:, :]

	df_train.to_csv(f_train, header=None, index=None)
	df_test.to_csv(f_test, header=None, index=None)

if __name__ == "__main__":
	main()