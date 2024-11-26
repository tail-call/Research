import torch


class MockCtx:
    """
    Mock version of torch context, for debuggning.
    """
    @property
    def saved_tensors(self):
        return (self.input, self.p)

    def save_for_backward(
        self,
        input: torch.Tensor,
        p: torch.Tensor,
    ):
        self.input = input
        self.p = p