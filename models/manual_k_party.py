import torch.nn as nn
from torchvision import models
import torch


class Manual_A(nn.Module):
    def __init__(self, u_dim=64, k=2):
        super(Manual_A, self).__init__()
        self.net = models.resnet18(pretrained=False, num_classes=u_dim)
        self.classifier = nn.Linear(u_dim, 40)
        self.contrastive_loss = 0
        self.T = 0.07
        self.register_buffer("queue", torch.randn(64, 1024))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        # ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, input, U_B):
        out = self.net(input)
        if U_B is not None:
            if not torch.sum(U_B[0]) == 0:
                l_pos = torch.einsum('nc,nc->n', [out, U_B[0]]).unsqueeze(-1)
                l_neg = torch.einsum('nc,ck->nk', [out, self.queue.clone().detach()])
                # apply temperature
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.T

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

                self.contrastive_loss = nn.CrossEntropyLoss().cuda()(logits, labels)
                self._dequeue_and_enqueue(U_B[0])
                # self.contrastive_loss = torch.sum((out - U_B[0])**2) / out.size(0)
                # self.contrastive_loss = torch.abs(torch.sum(out * U_B[0]) / out.size(0))
                # if torch.sum(U_B[0]) == 0:
                #     out = out
                # else:
                #     out = out * U_B[0]
                # out = out + U_B[0]
                # out = torch.cat([out] + [U for U in U_B], dim=1)
        logits = self.classifier(out)
        return logits, self.contrastive_loss

    def contrastive_loss(self):
        raise NotImplementedError


class Manual_B(nn.Module):

    def __init__(self, u_dim=64):
        super(Manual_B, self).__init__()
        self.net = models.resnet18(pretrained=False, num_classes=u_dim)

    def forward(self, input):
        u = self.net(input)
        return u

# class Manual_A(nn.Module):
#
#     def __init__(self, input_dim, u_dim=64, k=2):
#         super(Manual_A, self).__init__()
#         self.linear = nn.Linear(input_dim, u_dim)
#         self.classifier = nn.Linear(u_dim, 1)
#         self.dist_loss = 0
#
#     def forward(self, input, U_B):
#         out = self.linear(input)
#         if U_B is not None:
#             # self.dist_loss = torch.sum((out - U_B[0])**2) / out.size(0)
#             self.dist_loss = torch.abs(torch.sum(out * U_B[0]) / out.size(0))
#             out = out * U_B[0]
#             # out = out + U_B[0]
#             # out = torch.cat([out] + [U for U in U_B], dim=1)
#         logits = self.classifier(out)
#         return logits, self.dist_loss
#
#
# class Manual_B(nn.Module):
#
#     def __init__(self, input_dim, u_dim=64):
#         super(Manual_B, self).__init__()
#         self.linear = nn.Linear(input_dim, u_dim)
#
#     def forward(self, input):
#         out = self.linear(input)
#         return out
