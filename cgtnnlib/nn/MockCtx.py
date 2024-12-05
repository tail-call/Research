class MockCtx:
    """
    Mock version of torch context, for debuggning.
    """
    @property
    def saved_tensors(self):
        return (self.input, self.p)

    def save_for_backward(
        self,
        *args,
    ):
        self.saved_for_backward = args