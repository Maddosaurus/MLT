"""Generate distribution-related graphs as PNGs to disk."""
import os
import matplotlib.pyplot as plt
import pandas as pd

# See https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
# and https://stackoverflow.com/questions/18992086/save-a-pandas-series-histogram-plot-to-file
def generate_feature_distribution_to_disk(datas, title, resultpath, column_names=None):
    """Generate distribution graphs for a given dataset into the resultfolder"""
    if not isinstance(datas, pd.DataFrame):
        if not column_names:
            print("Warning! I don't have column names. Will use indices!")
        datas = pd.DataFrame(datas, columns=column_names)

    generate_boxplot_to_disk(datas, title, resultpath)
    generate_hist_to_disk(datas, title, resultpath)



def generate_boxplot_to_disk(data_pdframe, title, resultpath):
    """Generate a boxplot for given dataframe to the path"""
    # Prepare path
    savepath = os.path.join(resultpath, title+'_boxplot.png')

    # Generate the figure
    data_pdframe.boxplot(rot=90)
    plt.title('Boxplot for ' + title)
    plt.tight_layout()

    # Save fig and clear all caches
    plt.savefig(savepath)
    plt.clf()
    plt.close('all')


def generate_hist_to_disk(data_pdframe, title, resultpath):
    """Generate a histogram for given dataframe to the path"""
    # Prepare path
    savepath = os.path.join(resultpath, title+'_hist.png')

    # Generate the figure
    data_pdframe.hist(bins=30)
    plt.suptitle('Histogram for ' + title)

    # Add some spacing to the subplots
    plt.subplots_adjust(
        top=0.88, bottom=0.08,
        left=0.10, right=0.95,
        hspace=0.55, wspace=0.35
    )

    # Save fig and clear all caches
    plt.savefig(savepath)
    plt.clf()
    plt.close('all')
