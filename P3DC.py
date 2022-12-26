import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F



def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def knn_matrix(d, k):
    sort = np.argsort(d, axis=1)
    nn = sort[:, -k:]
    dd = np.zeros((n_ways * n_shot, R))
    for i in range(nn.shape[0]):
        for j in range(nn.shape[1]):
            dd[i, j] = d[i, nn[i, j]]
    dk = F.softmax(torch.Tensor(dd), dim=1)
    return dk, nn


if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 2000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples


    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)

    # ---- Base class statistics
    base_means = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset

    #加载特征
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            base_means.append(mean)

    base_prototype = np.array(base_means)
    # ---- classification for each task
    acc_list = []
    account = []
    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()

        support_data_norm = support_data / np.linalg.norm(support_data, 2, 1)[:, None]
        #变换
        beta = 0.5
        support_data = np.power(support_data[:, ], beta)

        R = 5
        #sample-level
        sw = np.dot(support_data, base_prototype.transpose())
        sbase, rn = knn_matrix(sw, R)
        sbase = np.expand_dims(sbase, 2).transpose(0, 2, 1)
        rneighbor = base_prototype[rn]
        sr = torch.bmm(torch.Tensor(sbase), torch.Tensor(rneighbor))
        sr = np.squeeze(sr.numpy())
        r = sr + support_data
        r = r / np.linalg.norm(r, 2, 1)[:, None]


        #task-level
        tw = np.dot(support_data, base_prototype.transpose())
        sort = np.argsort(tw, axis=1)
        index = sort[:, -R:]
        tneighbor = np.unique(index.ravel())
        tbase = base_prototype[tneighbor]
        ts = np.dot(support_data, tbase.transpose())
        ts = F.softmax(torch.Tensor(ts), dim=1)
        t = np.dot(ts, tbase) + support_data
        t = t/ np.linalg.norm(t, 2, 1)[:, None]

        a = 0.0
        b = 0.9
        c = 1 - a - b
        proto = c * support_data_norm + a * r + b * t
        proto = proto / np.linalg.norm(proto, 2, 1)[:, None]


        #NN分类
        query_data = query_data / np.linalg.norm(query_data, 2, 1)[:, None]
        predicts = np.dot(query_data, proto.transpose())
        predicts = np.squeeze(np.array(np.argmax(predicts, axis=1)))
        acc = np.mean(predicts== query_label)
        acc_list.append(acc)

    mean, confidence = compute_confidence_interval(acc_list)
    print('%s %d way %d shot  ACC : %f %f'%(dataset,n_ways,n_shot,float(mean), float(confidence)))


