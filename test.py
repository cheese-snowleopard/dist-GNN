from torch import tensor

node_dict = {'part_id': tensor([0, 0, 1, 0, 0, 1, 0, 1, 0])}
belong = (node_dict['part_id'] == 1)
print(belong.sum().view(-1))