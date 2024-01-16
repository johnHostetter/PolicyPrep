"""
Implements helpful functions to support Neural Architecture Search in D3RLPy.
"""
from typing import Dict

import torch
from d3rlpy.dataset import TransitionMiniBatch


def callback(algorithm, epoch, total_steps):
    if total_steps % 100 == 0:
        # temp = algorithm.impl._q_func.q_funcs[0]._encoder.flc.engine.temperature
        # algorithm.impl._q_func.q_funcs[
        #     0
        # ]._encoder.flc.engine.temperature = max(
        #     5 * (2 / epoch), 0.1
        # )
        # temp = torch.max(5 * (2 / epoch), 0.1)
        # state_dictionary = algorithm.impl._q_func.q_funcs[
        #     0
        # ]._encoder.flc.engine.state_dict()
        # state_dictionary["temperature"] = temp
        # algorithm.impl._q_func.q_funcs[
        #     0
        # ]._encoder.flc.engine.load_state_dict(state_dictionary)
        algorithm.impl._build_optim()
        print(
            total_steps,
            [
                mod.temperature
                for mod in algorithm.impl._q_func.q_funcs[
                    0
                ]._encoder.flc.engine.input_modules_list
            ],
        )


def new_update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
    if self._use_gpu:
        self.impl.to_gpu(self._use_gpu)
    else:
        self.impl.to_cpu()
    self.impl._build_optim()
    loss = self._update(batch)
    # print(loss.device)
    self._grad_step += 1
    # if (
    #     (
    #         torch.isclose(
    #             self.impl._q_func.q_funcs[0]
    #             ._encoder.flc.engine.input_modules_list[0]
    #             .logits.grad,
    #             torch.zeros(1),
    #         )
    #     )
    #     .all()
    #     .item()
    # ):
    #     print("ALL LOGITS HAVE ZERO GRAD?")
    return loss


# func_type = type(algorithm.update)
# algorithm.update = func_type(new_update, algorithm)

# algorithm.create_impl(
#     algorithm._process_observation_shape(
#         env.observation_space.shape
#     ),
#     env.action_space.n,
# )

# algorithm.inner_create_impl(
#     env.observation_space.shape, env.action_space.n
# )


def new_update_target(self):
    assert self.impl._q_func is not None
    assert self.impl._targ_q_func is not None
    print("in it")

    def my_sync(targ_model: torch.nn.Module, model: torch.nn.Module) -> None:
        # with torch.no_grad():
        params = model.named_parameters()
        targ_params = targ_model.named_parameters()
        dict_targ_params = dict(targ_params)
        for name, param in params:
            if name in targ_params:
                dict_targ_params[name].data.copy_(param.data)

    try:
        my_sync(self.impl._targ_q_func, self.impl._q_func)
    except RuntimeError:
        from copy import deepcopy

        print("copying")
        # with torch.no_grad():
        # self._targ_q_func = deepcopy(self._q_func)
        weights_path = "weights_tmp.pt"
        torch.save(self._q_func.state_dict(), weights_path)
        self._targ_q_func = torch.load(weights_path)


# func_type = type(algorithm.impl.update_target)

# algorithm.impl.update_target = func_type(
#     new_update_target, algorithm.impl
# )


def modify_algorithm(algorithm, num_of_features: int, num_of_actions: int):
    func_type = type(algorithm.update)
    algorithm.update = func_type(new_update, algorithm)

    algorithm.create_impl(
        algorithm._process_observation_shape((num_of_features,)),
        num_of_actions,
    )
    func_type = type(algorithm.impl.update_target)
    algorithm.impl.update_target = func_type(new_update_target, algorithm)

    return algorithm
