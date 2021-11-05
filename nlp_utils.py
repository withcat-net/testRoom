import os
import pyhdfs
import logging
import urllib.request

class TM:
    def __init__(self, **kwargs):
        if 'pm' in kwargs:
            from preprocess_run import process_for_train
            pm = kwargs.pop('pm', None)
            process_for_train(pm)
            self.train_data_path=pm.target_path
        else:
            self.train_data_path='.'
        self.param_info = kwargs

from pandas import DataFrame as DF
class PM:
    def __init__(self,source_path='/home/ubuntu/jwchoi/NLP/ocr_data/source'):
        self.source_path=source_path
        self.target_path=f'{source_path.rstrip("/")}_target'

def get_file_path(file_url, save_dir='/work/hdfs_data'):
    if '://' not in file_url:
        return file_url
    url_download_prefix='urllib'
    split_list = file_url.split('/')
    local_path = f'{save_dir}/{"/".join(split_list[2:])}'
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        assert not file_url.startswith(url_download_prefix)
        client = pyhdfs.HdfsClient('/'.join(split_list[:3]))
        exists = client.exists(file_url)
        logging.info(exists)
        if not exists:
            raise ValueError(f'file not exists: {file_url}')
        client.copy_to_local(file_url, local_path)
    except:
        if file_url.startswith(url_download_prefix):
            file_url=file_url.lstrip(url_download_prefix)
        urllib.request.urlretrieve(file_url, local_path)
    return local_path

IS_PLATFORM=True