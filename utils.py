def get_id2url_map():
    import urllib
    
    id2url = dict()
    with open('./artwork_ids.txt') as f:
        for line in f.readlines():
            line = line.rstrip()
            _id, url = line.split(' ', 1)
            assert(url.index(_id) > 0)
            _id = int(_id)
            if url.index('static') == 0:
                url = 'http://' + url
            idx = url.index('Images/') + len('Images/')
            url = url[:idx] + urllib.parse.quote(url[idx:])
            id2url[_id] = url
    return id2url

def silentremove(filename):
    import os
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def download_image(url, outpath):
    import requests
    import shutil
    try:
        r = requests.get(url, stream=True, timeout=3)
        if r.status_code == 200:
            with open(outpath, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        return True
    except requests.exceptions.Timeout as e:
        print('Timeout detected: url=',url)
        print(e)
        silentremove(outpath)
    except Exception as e:
        print('Unexpected exception detected: url=',url)
        print(e)
        silentremove(outpath)
    finally:
        r.close()
    return False

def process_image_batch(
        models, featmats, image_files, image_ids, id2url_map,
        i_start, i_end, preprocess_input_fn, image_target_size=(224, 224)):
    assert len(models) > 0 and len(models) == len(featmats)
    assert 0 <= i_start < i_end <= len(image_files)
    
    import numpy as np
    from keras.preprocessing import image
    
    n = i_end - i_start
    batch_X = np.empty(shape=(n, *image_target_size, 3))
    for i in range(i_start, i_end):
        file = image_files[i]
        img_loaded = False
        tries = 0
        while 1:
            try:
                img = image.load_img(file, target_size=image_target_size)
                img_loaded = True
                break
            except OSError as e:
                print('OSError detected, file = ', file)                
                tries += 1                
                if (tries == 3):
                    break
                url = id2url_map[image_ids[i]]
                print('(attempt %d) we will try to download the image from url=%s' % (tries, url))
                if download_image(url, file):
                    print('image successfully downloaded to %s!' % file)
        if not img_loaded:            
            raise Exception('failed to load image file=%s, url=%s' % (file, url))
        batch_X[i - i_start] = image.img_to_array(img)
    batch_X = preprocess_input_fn(batch_X)
    for model, featmat in zip(models, featmats):
        featmat[i_start:i_end] = model.predict(batch_X)
        
def get_image(image_cache, _id):
    try:
        return image_cache[_id]
    except KeyError:
        from PIL import Image
        img = Image.open('/mnt/workspace/Ugallery/images/%d.jpg' % _id)
        image_cache[_id] = img
        return img

def plot_images(plt, image_cache, ids):    
    plt.close()
    n = len(ids)    
    nrows = n//5 + int(n%5>0)
    ncols = min(n, 5)
    plt.figure(1, (20, 5 * nrows))
    for i, _id in enumerate(ids):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        img = get_image(image_cache, _id)
        ax.set_title('%d) id = %d' % (i, _id))
        ax.imshow(img, interpolation="nearest")
    plt.show()
    
    
def read_ids_file(dirpath, ids_filename):
    from os import path    
    filepath = path.join(dirpath, ids_filename)
    if ids_filename[-5:] == '.json':
        with open(filepath) as f:
            index2id = json.load(f)
    elif ids_filename[-4:] == '.npy':
        import numpy as np
        index2id = np.load(filepath)
    else:
        assert ids_filename[-3:] == 'ids'
        with open(filepath) as f:
            index2id = [int(x) for x in f.readlines()]
    id2index = {_id:i for i, _id in enumerate(index2id)}
    return index2id, id2index

def load_embeddings_and_ids(dirpath, embedding_file, ids_file):
    import numpy as np
    from os import path
    featmat = np.load(path.join(dirpath, embedding_file))
    index2id, id2index = read_ids_file(dirpath, ids_file)
    return dict(
        featmat=featmat,
        index2id=index2id,
        id2index=id2index,
    )

def concatenate_featmats(artwork_ids, featmat_list, id2index_list):    
    assert len(featmat_list) == len(id2index_list)
    import numpy as np
    n = len(artwork_ids)
    m = sum(fm.shape[1] for fm in featmat_list)
    out_mat = np.empty(shape=(n,m))
    for i, _id in enumerate(artwork_ids):
        out_mat[i] = np.concatenate(
            [fm[id2index[_id]] for fm, id2index in zip(featmat_list, id2index_list)])
    return out_mat

class User:
    def __init__(self, uid):
        self._uid = uid
        self.artwork_ids = []
        self.artwork_idxs = []
        self.artwork_idxs_set = set()
        self.timestamps = []
        self.artist_ids_set = set()
        self.cluster_ids_set = set()
        
    def clear(self):
        self.artwork_ids.clear()
        self.artwork_idxs.clear()
        self.artwork_idxs_set.clear()        
        self.artist_ids_set.clear()
        self.cluster_ids_set.clear()
        self.timestamps.clear()
    
    def refresh_nonpurchased_cluster_ids(self, n_clusters):
        self.nonp_cluster_ids = [c for c in range(n_clusters) if c not in self.cluster_ids_set]
        assert len(self.nonp_cluster_ids) > 0
        
    def refresh_cluster_ids(self):
        self.cluster_ids = list(self.cluster_ids_set)
        assert len(self.cluster_ids) > 0
        
    def refresh_artist_ids(self):
        self.artist_ids = list(self.artist_ids_set)
        assert len(self.artist_ids) > 0
        
    def append_transaction(self, artwork_id, timestamp, artwork_id2index, artist_ids, cluster_ids):
        aidx = artwork_id2index[artwork_id]
        self.artwork_ids.append(artwork_id)
        self.artwork_idxs.append(aidx)
        self.artwork_idxs_set.add(aidx)
        self.artist_ids_set.add(artist_ids[aidx])
        self.cluster_ids_set.add(cluster_ids[aidx])
        self.timestamps.append(timestamp)
    
    def remove_last_nonfirst_purchase_basket(self, artwork_id2index, artist_ids, cluster_ids):
        baskets = self.baskets
        len_before = len(baskets)
        if len_before >= 2:
            last_b = baskets.pop()
            artwork_ids = self.artwork_ids[:last_b[0]]
            timestamps = self.timestamps[:last_b[0]]
            self.clear()
            for aid, t in zip(artwork_ids, timestamps):
                self.append_transaction(aid, t, artwork_id2index, artist_ids, cluster_ids)
            assert len(self.baskets) == len_before - 1
        
    def build_purchase_baskets(self):
        baskets = []
        prev_t = None
        offset = 0
        count = 0
        for i, t in enumerate(self.timestamps):
            if t != prev_t:
                if prev_t is not None:
                    baskets.append((offset, count))
                    offset = i
                count = 1
            else:
                count += 1
            prev_t = t
        baskets.append((offset, count))
        self.baskets = baskets
        
    def sanity_check_purchase_baskets(self):
        ids = self.artwork_ids
        ts = self.timestamps
        baskets = self.baskets        
        n = len(ts)
        assert(len(ids) == len(ts))
        assert(len(baskets) > 0)
        assert (n > 0)
        for b in baskets:
            for j in range(b[0], b[0] + b[1] - 1):
                assert(ts[j] == ts[j+1])
        for i in range(1, len(baskets)):
            b1 = baskets[i-1]
            b2 = baskets[i]
            assert(b1[0] + b1[1] == b2[0])
        assert(baskets[0][0] == 0)
        assert(baskets[-1][0] + baskets[-1][1] == n)
        
class VisualDuplicateDetector:
    def __init__(self, cluster_ids, embeddings):        
        self._cluster_ids = cluster_ids
        self._embeddings = embeddings
        self._equalityCache = dict()
        self.count = 0
        
    def same(self,i,j):
        if self._cluster_ids[i] != self._cluster_ids[j]:
            return False
        if i > j:
            i, j = j, i
        k = (i,j)
        try:
            ans = self._equalityCache[k]
        except KeyError:
            from numpy import array_equal
            ans = self._equalityCache[k] = array_equal(self._embeddings[i], self._embeddings[j])
        if ans:
            self.count += 1
        return ans
    
def get_decaying_learning_rates(maxlr, minlr, decay_coef):
    assert maxlr > minlr > 0
    assert 0 < decay_coef < 1
    lrs = []
    lr = maxlr
    while lr >= minlr:
        lrs.append(lr)
        lr *= decay_coef
    return lrs