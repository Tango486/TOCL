import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase.
"""







class Categorical_Topk(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(Categorical_Topk, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.K = args.use_K

    def forward(self, x, available_actions=None):
        x = self.linear(x)

        if available_actions is not None:
            x[available_actions == 0] = -1e10

        self.probs = torch.softmax(x, dim=-1)
        self.x = x
        return x

    def sample(self):
        return torch.multinomial(self.probs, num_samples=self.K, replacement=False)

    def log_probs(self, actions):
        return torch.gather(torch.log(self.probs), 0, actions.long())

    def mode(self):

        return torch.topk(self.probs, self.K, dim=-1)[1]


    def entropy(self):
        return FixedCategorical_Topk(logits=self.x).entropy()


class FixedCategorical_Topk(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None, active_masks=None):
        """
        前向传播

        Args:
            x: (batch_size, feat_dim) 输入特征
            available_actions: (batch_size, num_actions) 可用动作mask，0表示不可用
            active_masks: (batch_size,) bool tensor，标识有效的agents

        Returns:
            FixedCategorical: 分类分布对象
        """

        x = self.linear(x)


        if available_actions is not None:
            x[available_actions == 0] = -1e10



        if active_masks is not None:

            x[~active_masks] = -1e10

            x[~active_masks, 0] = 0.0

        return FixedCategorical(logits=x, active_masks=active_masks)


class FixedCategorical(torch.distributions.Categorical):
    """
    固定的分类分布，支持 active_masks 来处理 padding agents
    """
    def __init__(self, probs=None, logits=None, validate_args=None, active_masks=None):
        """
        Args:
            probs: 概率分布
            logits: logits
            validate_args: 是否验证参数
            active_masks: (batch_size,) bool tensor，标识有效的 agents
        """
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        self.active_masks = active_masks

    def sample(self):
        """
        采样动作
        对于 padding agents (active_masks=False)，返回默认动作 0

        Returns:
            actions: (batch_size, 1)
        """
        samples = super().sample().unsqueeze(-1)


        if self.active_masks is not None:
            samples = samples.clone()
            samples[~self.active_masks] = 0

        return samples

    def log_probs(self, actions):
        """
        计算动作的 log 概率
        对于 padding agents，返回 0.0（不贡献梯度）

        Args:
            actions: (batch_size, 1)
        Returns:
            log_probs: (batch_size, 1)
        """
        log_probs = (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )


        if self.active_masks is not None:
            log_probs = log_probs * self.active_masks.unsqueeze(-1).float()

        return log_probs

    def mode(self):
        """
        返回最可能的动作（贪婪策略）
        对于 padding agents，返回默认动作 0

        Returns:
            actions: (batch_size, 1)
        """
        modes = self.probs.argmax(dim=-1, keepdim=True)


        if self.active_masks is not None:
            modes = modes.clone()
            modes[~self.active_masks] = 0

        return modes


































class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean



class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()




class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)


        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
