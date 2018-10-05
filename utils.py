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