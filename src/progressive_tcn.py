"""Progressive fine-tuning of LiteTCN-R; the linear/RevIN baseline stays fixed."""
import torch

from .residual_tcn import ResidualLiteTCN


class ProgressiveLiteTCN(ResidualLiteTCN):
    def set_stage(self, mode, round_number):
        if mode not in ("head_only", "progressive"):
            raise ValueError(mode)
        stage = "head"
        if mode == "progressive" and round_number > 4:
            stage = "projection"
        if mode == "progressive" and round_number > 8:
            stage = "encoder"
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith("head.") or
                             (stage in ("projection", "encoder") and name.startswith("proj.")) or
                             (stage == "encoder" and name.startswith("enc.")))
        # Equal deterministic feature computation in both controls. eval() does not
        # prevent gradients; disabling dropout avoids confounding unfreezing with noise.
        self.proj[1].eval()
        return stage

    def optimizer_groups(self):
        return [{"params": list(self.head.parameters()), "lr": 3e-4},
                {"params": list(self.proj.parameters()), "lr": 3e-5},
                {"params": list(self.enc.parameters()), "lr": 1e-5}]


def forecast_objective(prediction, truth, key):
    error = (prediction-truth).square()
    return .5*error.mean()+.5*error[..., key].mean()


def relative_anchor(model, reference):
    terms = [(p-reference[n]).square().mean() / (reference[n].square().mean()+1e-4)
             for n, p in model.named_parameters() if p.requires_grad]
    return torch.stack(terms).mean()


def input_pair(x, horizon):
    """One isolated input window -> prefix with left padding, withheld observed tail.

    The withheld suffix never enters the predictor, even through normalization.
    No overlapping windows are joined and no target future tensor is accepted.
    """
    if horizon < 1 or x.shape[1] <= horizon:
        raise ValueError("Need a positive horizon smaller than the input")
    prefix = x[:, :-horizon]
    return torch.cat((prefix[:, :1].expand(-1, horizon, -1), prefix), 1), x[:, -horizon:].detach()
