import torch


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, features, head, queue=None):
        if queue is None or queue.shape[0] == 0:
            queue = None
        if queue is not None:
            features = torch.vstack((features, queue))

        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=1, p=2)

        logits = features@head

        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        to_ret = Q.t() if queue is None else Q.t()[:-queue.shape[0]]

        return to_ret