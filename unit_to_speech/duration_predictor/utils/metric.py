import torch

class Accuracy:
    def __init__(self):
        self.y, self.yhat = [], []

    def update(self, yhat, y):
        self.yhat.append(yhat)
        self.y.append(y)

    def acc(self, tol):
        yhat = torch.cat(self.yhat)
        y = torch.cat(self.y)
        acc = torch.abs(yhat - y) <= tol
        acc = acc.float().mean().item()
        return acc