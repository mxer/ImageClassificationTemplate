import torch
from collections import defaultdict

""" if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu') """

# for binary classification
def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()

def accuracy(y_hat, y_true):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y_true.dtype) == y_true
    return float(cmp.type(y_true.dtype).sum())


def evaluate_accuracy(model, data_iter):
    if isinstance(model, torch.nn.Module):
        model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y, img_path in data_iter:
            y_pred = model(X.cuda())
            metric.add(accuracy(y_pred, y.long().cuda()), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n  # n个参数要记录

    def add(self, *args):
        # print(args)
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )