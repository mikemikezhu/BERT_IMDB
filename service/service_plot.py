import matplotlib.pyplot as plt
import seaborn as sns


class PlotService:

    """ Public methods """

    def plot_hist(self, plot_param):

        plt.title(plot_param.plot_title)
        plt.plot(plot_param.train_hist, label=plot_param.train_label)
        plt.plot(plot_param.val_hist, label=plot_param.val_label)
        plt.xlabel(plot_param.x_label)
        plt.ylabel(plot_param.y_label)
        plt.savefig(plot_param.file_name)

    def plot_attention(self, plot_param):

        plt.title(plot_param.plot_title)
        sns.heatmap(plot_param.attention)
        plt.savefig(plot_param.file_name)
