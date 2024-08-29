import matplotlib.pyplot as plt
import numpy as np
import shap


def get_shap_values(model, background, test_images, check_additivity=False):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_images, check_additivity=check_additivity)
    return shap_values


def plot_explanation(shap_values, test_images, gt_label, fig_title="Backbone explanation with Deep SHAP", labels=["NON MS", "MS"], path="./"):

    test_numpy = test_images.squeeze().cpu().numpy()
    test_numpy = test_numpy[..., np.newaxis]

    for i in range(shap_values[0].shape[2]):
        shap_numpy = []
        # 3,1,50,50,50 -> 3,50,50,50
        # batch, channel, depth, height, width -> depth, height, width, channel

        shap_array = shap_values[0].squeeze(axis=0)[:, i, :, :]
        shap_array = shap_array[..., np.newaxis]
        shap_numpy.append(shap_array)
        shap_array = shap_values[1].squeeze(axis=0)[:, i, :, :]
        shap_array = shap_array[..., np.newaxis]
        shap_numpy.append(shap_array)
        print(i)
        shap.image_plot(shap_numpy, (test_numpy[i, :, :, :])[np.newaxis, ...],
                        show=False, aspect=1, labels=labels)
        figure = plt.gcf()

        figure.suptitle(fig_title)
        ax = figure.get_axes()
        ax[0].set_title(gt_label)
        # plt.show()
        figure.savefig(f"{path}/figurename{i}.png")
        plt.close()

