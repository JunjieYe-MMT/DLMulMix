


data_idx_path = r'data-bin/en-vi/train.en-vi.en.idx'
data_bin_path = r'data-bin/en-vi/train.en-vi.en.bin'

with open(data_idx_path, 'r', encoding='unicode-escape') as fi:
    data_idx = fi.read()

with open(data_bin_path, 'rb') as fb:
    data_idx = fb.readlines()


123