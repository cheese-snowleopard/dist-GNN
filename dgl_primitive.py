import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import RedditDataset
from dgl.nn import SAGEConv

# Load Reddit dataset
data = RedditDataset(self_loop=True)
g = data[0]
train_mask = g.ndata['train_mask']
labels = g.ndata['label']

# Define GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Create model, optimizer and loss function
model = GraphSAGE(g.ndata['feat'].shape[1], 16, data.num_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fcn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    logits = model(g, g.ndata['feat'])
    loss = loss_fcn(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
