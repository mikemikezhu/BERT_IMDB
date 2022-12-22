import matplotlib.pyplot as plt


class PlotService:

    """ Public methods """

    def plot_hist(self, plot_query):

        plt.title(plot_query.plot_title)
        plt.plot(plot_query.train_hist, label=plot_query.train_label)
        plt.plot(plot_query.val_hist, label=plot_query.val_label)
        plt.xlabel(plot_query.x_label)
        plt.ylabel(plot_query.y_label)
        plt.savefig(plot_query.file_name)
