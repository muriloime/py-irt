# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# pylint: disable=unused-argument,unused-variable,not-callable,no-name-in-module,no-member,protected-access
from functools import partial
from py_irt.models import abstract_model

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import SVI, EmpiricalMarginal, Trace_ELBO
from pyro.optim import Adam
from rich.console import Console
from rich.live import Live
from rich.table import Table
import numpy as np

console = Console()


@abstract_model.IrtModel.register("3pl")
class ThreeParamLog(abstract_model.IrtModel):
  """3PL IRT Model"""

  # pylint: disable=not-callable
  def __init__(self,
               *,
               num_items: int,
               num_subjects: int,
               verbose=False,
               device: str = "cuda",
               **kwargs
               ):
    super().__init__(
        num_items=num_items, num_subjects=num_subjects, device=device, verbose=verbose
    )
    self.params = {}
    self.params['a'] = torch.ones(self.num_items, device=self.device) * 0.5
    self.params['a_std'] = torch.ones(self.num_items, device=self.device) * 0.3
    self.params['b'] = torch.ones(self.num_items, device=self.device) * 1.0
    self.params['b_std'] = torch.ones(self.num_items, device=self.device) * 1.0
    self.params['c'] = torch.ones(self.num_items, device=self.device) * 0.2
    self.params['c_std'] = torch.ones(self.num_items, device=self.device) * .06
    self.params['alpha_c'] = torch.ones(self.num_items, device=self.device) * 2
    self.params['beta_c'] = torch.ones(self.num_items, device=self.device) * 17

  def model_hierarchical(self, models, items, obs):
    # Theta param
    m_theta = torch.zeros(self.num_subjects, device=self.device)
    s_theta = torch.ones(self.num_subjects, device=self.device)
    with pyro.plate("thetas", self.num_subjects, device=self.device):
      ability = pyro.sample("theta", dist.Normal(m_theta, s_theta))

    with pyro.plate("params", self.num_items, device=self.device):
      # Discrimination param
      disc = pyro.sample("a", dist.LogNormal(
          self.params['a'], self.params['a_std']))
      diff = pyro.sample("b", dist.Normal(
          self.params['b'], self.params['b_std']))
      lambdas = pyro.sample('c', dist.Beta(
          self.params['alpha_c'], self.params['beta_c']))

    # import pdb
    # pdb.set_trace()
    p_star = lambdas[items] + (1-lambdas[items]) * \
        torch.sigmoid(disc[items] * (ability[models] - diff[items]))

    with pyro.plate("observe_data", obs.size(0)):
      pyro.sample("obs", dist.Bernoulli(probs=p_star), obs=obs)

  def guide_hierarchical(self, models, items, obs):
    # theta param
    m_theta_param = pyro.param(
        "loc_ability", torch.zeros(self.num_subjects, device=self.device)
    )
    s_theta_param = pyro.param(
        "scale_ability",
        torch.ones(self.num_subjects, device=self.device),
        constraint=constraints.positive,
    )
    with pyro.plate("thetas", self.num_subjects, device=self.device):
      pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

    # sample discrimitation params (disc)

    m_a_param = pyro.param(
        "loc_disc",
        # self.params['a'],
        lambda: torch.empty(self.num_items, device=self.device).fill_(1.0),
        constraint=constraints.interval(0.2, 6.499)
    )

    # Difficulty param
    m_b_param = pyro.param(
        "loc_diff",
        # self.params['b'],
        lambda: torch.empty(self.num_items, device=self.device).fill_(1.0),
        constraint=constraints.interval(-3.0, 5.0))
    # sample discrimitation params (disc)
    m_c_param = pyro.param(
        "loc_guess",
        # self.params['c'],
        lambda: torch.empty(self.num_items, device=self.device).fill_(0.15),
        constraint=constraints.interval(0.001, 0.499))

    with pyro.plate("params", self.num_items, device=self.device):
      disc = pyro.sample('a', dist.Delta(m_a_param))
      diff = pyro.sample("b", dist.Delta(m_b_param))
      lambdas = pyro.sample('c', dist.Delta(m_c_param))

  def export(self):
    return {name: pyro.param(name).data.tolist() for name in pyro.get_param_store().get_all_param_names()}
    return {
        # "ability": pyro.param("loc_ability").data.tolist(),
        # "scale_ability": pyro.param("scale_ability").data.tolist(),
        "diff": pyro.param("loc_diff").data.tolist(),
        # "scale_diff": pyro.param("scale_diff").data.tolist(),
        # "a": torch.exp(pyro.param("loc_disc") + pyro.param("scale_disc") ** 2).data.tolist(),
        "disc": pyro.param("loc_disc").data.tolist(),
        "guess": pyro.param("loc_guess").data.tolist(),
        # "scale_disc": pyro.param("scale_disc").data.tolist(),
        # "lambdas": (pyro.param("alpha_c")/(pyro.param("alpha_c") + pyro.param("beta_c"))).data.tolist(),
    }

  def predict(self, subjects, items, params_from_file=None):
    """predict p(correct | params) for a specified list of model, item pairs"""
    if params_from_file is not None:
      model_params = params_from_file
    else:
      model_params = self.export()
    abilities = np.array([model_params["ability"][i] for i in subjects])
    diffs = np.array([model_params["diff"][i] for i in items])
    discs = np.array([model_params["disc"][i] for i in items])
    lambdas = np.array([model_params["lambdas"][i] for i in items])
    return lambdas + (1 - lambdas) / (1 + np.exp(-discs * (abilities - diffs)))

  def get_guide(self):
    # return pyro.infer.autoguide.AutoDelta(self.get_model())
    return self.guide_hierarchical

  def get_model(self):
    return self.model_hierarchical

  def summary(self, traces, sites):
    marginal = (
        EmpiricalMarginal(traces, sites)._get_samples_and_weights()[
            0].detach().cpu().numpy()
    )
    print(marginal)
    site_stats = {}
    for i in range(marginal.shape[1]):
      site_name = sites[i]
      marginal_site = pd.DataFrame(marginal[:, i]).transpose()
      describe = partial(pd.Series.describe, percentiles=[
          0.05, 0.25, 0.5, 0.75, 0.95])
      site_stats[site_name] = marginal_site.apply(describe, axis=1)[
          ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
      ]
    return site_stats
