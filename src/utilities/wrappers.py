import torch


class TimeDistributed(torch.nn.Module):
    """
    A wrapper class for PyTorch modules that allows them to operate on a sequence of data.
    """

    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module: torch.nn.Module, batch_first: bool = False):
        """
        Initialize the TimeDistributed wrapper class.

        Args:
            module: A PyTorch module.
            batch_first: Whether the batch dimension is the first dimension.
        """
        super().__init__()
        self.module = module
        # self.modules_list = torch.nn.ModuleList([])
        # for idx in range(12):
        #     from copy import deepcopy
        #     self.modules_list.add_module(str(idx), deepcopy(module))
        self.batch_first = batch_first

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TimeDistributed wrapper class.

        Args:
            input_data: The input data.

        Returns:
            The output of the module on the input sequence of data.
        """
        if len(input_data.size()) <= 2:
            return self.module(input_data)

        # res = []
        # for idx, mod in enumerate(self.modules_list):
        #     dat = input_data[:, idx, :]
        #     # print(dat.shape)
        #     # print(dat)
        #     res.append(mod(dat))
        #
        # return torch.stack(res).transpose(0, 1)

        # squash samples and timesteps into a single axis
        reshaped_input_data = input_data.contiguous().view(
            -1, input_data.size(-1)
        )  # (samples * timesteps, input_size)

        module_output = self.module(reshaped_input_data)

        # reshape the output back to the original shape
        output_dim = 1
        if module_output.ndim == 2:
            output_dim = module_output.size(-1)
        if self.batch_first:
            module_output = module_output.contiguous().view(
                input_data.size(0), input_data.size(1), output_dim
            )  # (samples, timesteps, output_size)
        else:
            module_output = module_output.view(
                input_data.size(1), input_data.size(0), output_dim
            )  # (timesteps, samples, output_size)

        return module_output
