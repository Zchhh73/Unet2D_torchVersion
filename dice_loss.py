import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        # 利用预测值与标签相乘作交集
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score

# loss = DiceLoss()
# predict = torch.randn(3, 4, 4)
# target = torch.randn(3, 4, 4)
# score = loss(predict, target)
# print(predict)
# print(target)
# print(score)
