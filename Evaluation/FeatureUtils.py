import json
import os.path as ospath
import numpy as np

def read_ids_file(dirpath, ids_filename):
    filepath = ospath.join(dirpath, ids_filename)
    if ids_filename[-5:] == '.json':
        with open(filepath) as f:
            index2id = json.load(f)
    elif ids_filename[-4:] == '.npy':        
        index2id = np.load(filepath)
    else:
        assert ids_filename[-3:] == 'ids'
        with open(filepath) as f:
            index2id = [int(x) for x in f.readlines()]
    id2index = {_id:i for i, _id in enumerate(index2id)}
    return index2id, id2index

def read_numpy_features_matrix(dirpath, feat_filename, ids_filename):
    features = np.load(ospath.join(dirpath, feat_filename))
    index2id, id2index = read_ids_file(dirpath, ids_filename)
    return dict(
        features=features,
        index2id=index2id,
        id2index=id2index,
    )