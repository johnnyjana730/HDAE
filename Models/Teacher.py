import torch

class Teacher(torch.nn.Module):
    def __init__(self, num_aspect, general_aspect) -> None:
        super(Teacher, self).__init__()
        self.num_aspect = num_aspect
        self.general_aspect = general_aspect
        
    def forward(self, bow, zs):
        """Teacher
        Args:
            bow (torch.tensor): [B, bow_size]
            zs  (torch.tensor): [num_asp, bow_size]
        Returns:
            : [B, num_aspect]
        """

        result = [
            (bow[:, i, :] * zs[i, :].unsqueeze(0)).sum(-1).unsqueeze(-1)
            for i in range(self.num_aspect)
        ]
        result = torch.cat(result, -1)
        mask = bow.sum(-1).sum(-1) == 0
        result[mask, self.general_aspect] = 1
        return torch.softmax(result, -1)