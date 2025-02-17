import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, 256)

        self.post_net = nn.Linear(256, output_dim)
        
        self.activate = nn.ReLU()

    def forward(self, x, padding_mask=None):
        x = self.activate(self.pre_net(x))

        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)  # Compute average
        
        x = self.post_net(x)
        return x

# if __name__ == '__main__':
#     label_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
#     idx2label = {v: k for k, v in label_dict.items()}
#     model = BaseModel(input_dim=768, output_dim=len(label_dict))
#
#     ckpt = torch.load('C:/Users/ROG/Desktop/project/emotion2vec/emotion2vec_base/emotion2vec_base.pt')
#     model.load_state_dict(ckpt)
    #
    # feat = torch.randn(1, 100, 768)
    # padding_mask = torch.zeros(1, 100).bool()
    # outputs = model(feat, padding_mask)
    #
    # _, predict = torch.max(outputs.data, dim=1)
    # print(idx2label[predict.item()])