import torch
import numpy as np

src_len = [94, 8, 11, 11, 14, 29, 21, 13, 6, 7, 6, 9, 11, 10, 34, 21, 25, 24]
graph_len = [93, 7, 10, 10, 16, 31, 20, 12, 5, 6, 5, 8, 10, 9, 33, 20, 24, 23]
src_len_tensor = torch.tensor(src_len)

# max_len = max(src_len)

src_graph = []
for i in range(18):
    src_graph.append(np.random.randn(graph_len[i], graph_len[i]))

def collate_graph_tokens(values, src_len_tensor, pad_idx):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(src_len_tensor).item()
    res = pad_idx*torch.ones([len(values), size, size])
    # res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        tmp_src_len = src_len_tensor[i].item()
        v = torch.tensor(v)
        if tmp_src_len == size and len(v) >= tmp_src_len:
            copy_tensor(v[0:v.shape[0],0:v.shape[0]], res[i])
        else:
            copy_tensor(v, res[i][1:len(v) + 1, 1:len(v) + 1])
        # if len(v)<tmp_src_len:
        #     copy_tensor(v, res[i][1:len(v)+1,1:len(v)+1])
        # elif len(v)>tmp_src_len:
        #     copy_tensor(v, res[i][1:len(v) + 1])

    return res



rrreess = collate_graph_tokens(src_graph, src_len_tensor, 0)

123
