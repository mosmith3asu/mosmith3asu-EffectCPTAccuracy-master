import numpy as np

default_dir = r'C:\Users\mason\Desktop\EffectsCPTAccuracy-master\learning'


def save(data,filename,dir=None,method = 'numpy'):
    """ :param data: dict if using numpy """
    if dir is None: dir = default_dir
    full_name = dir + '\\' + filename
    print(f'\nSaving "{full_name}"... ',end='')
    if method=='numpy':
        if '.npz' not in filename: filename += '.npz'
        np.savez_compressed(full_name,**data)
        print(f'[DONE]')

def load(filename,dir=None,method = 'numpy'):
    """ :param data: dict if using numpy """
    result = {}
    if dir is None: dir = default_dir
    full_name = dir+'\\'+filename
    print(f'\nLoading "{full_name}"... ')
    if method=='numpy':
        if '.npz' not in filename: filename += '.npz'
        loaded = np.load(full_name)
        for key in loaded:
            print(f'\t| loading {key}...', end='')
            print(f'\r\t| {key}: {np.shape(loaded[key])}')
            result[key] = loaded[key] # load to memory
        print(f'\t| [DONE]')
        return result


if __name__ == "__main__":
    load('MDP_W1.npz')