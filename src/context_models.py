"""Linear branch uses L-H real context values during both source and pretext fitting."""
from .refined_models import ChannelRidge
from .strict_models import StrictEnsemble, input_pretext


class ContextRidge(ChannelRidge):
    def __init__(self, L, H, C):
        self.context = L - H
        super().__init__(self.context, H, C)
        self.L = L

    def forward(self, x):
        return super().forward(x[:, -self.context:])


class ContextEnsemble(StrictEnsemble):
    def __init__(self, cfg, neural_spec):
        super().__init__(cfg, neural_spec)
        for name in ("linear_source", "linear_adapted"):
            self.models[name] = ContextRidge(self.L, self.H, self.C)


def fit_context_source(cfg, source, out):
    from .refinement import fit_source
    context = cfg["L"] - cfg["H"]
    shorter = {k: (source[k][0][:, -context:], source[k][1]) for k in ("train", "val")}
    fitted, choices, statistics = fit_source({**cfg, "L": context}, shorter, out)
    result = ContextRidge(cfg["L"], cfg["H"], len(cfg["columns"]))
    result.load_state_dict(fitted.state_dict())
    return result, choices, statistics


def real_context_pretext(inputs, horizon):
    # Strictly INSIDE each original window. There is no need for left padding
    # because the linear forecast function always uses L-H recent values.
    return inputs[:, :-horizon], inputs[:, -horizon:].detach()
