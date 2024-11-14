import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def t_sne(embeds, labels, sample_num=2000, show_fig=True):
    sample_index = np.random.randint(0, embeds.shape[0], sample_num)  # 表示随机采样sample_num个
    sample_embeds = embeds[sample_index]
    sample_labels = labels[sample_index]

    ts = TSNE(n_components=2, init='pca', random_state=0)  # 降维，降到两个维度
    ts_embeds = ts.fit_transform(sample_embeds[:, :])

    mean, std = np.mean(ts_embeds, axis=0), np.std(ts_embeds, axis=0)

    for i in range(len(ts_embeds)):
        if (ts_embeds[i] - mean < 3 * std).all():
            np.delete(ts_embeds, i)

    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    fig = plt.figure()
    for i in range(norm_ts_embeds.shape[0]):
        plt.text(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1], str(sample_labels[i]),
                 color=plt.cm.Set1(sample_labels[i] % 7),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE', fontsize=14)
    plt.axis('off')
    if show_fig:
        plt.show()
    return fig


def similarity_plot(embedding, label, sample_num=1000, show_fig=True):
    label_sample = label[:sample_num]
    embedding_sample = embedding[:sample_num, :]

    cat = np.concatenate([embedding_sample, label_sample.reshape(-1, 1)], axis=1)
    arg_sort = np.argsort(label_sample)
    cat = cat[arg_sort]
    embedding_sample = cat[:, :-1]

    norm_embedding_sample = embedding_sample / np.sqrt(np.sum(embedding_sample ** 2, axis=1)).reshape(-1, 1)
    cosine_sim = np.matmul(norm_embedding_sample, norm_embedding_sample.transpose())
    cosine_sim[cosine_sim < 1e-5] = 0

    fig = plt.figure()
    sns.heatmap(data=cosine_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.axis("off")

    if show_fig:
        plt.show()
    return fig
