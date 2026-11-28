import torch


@torch.no_grad()
def gae(reward, value, next_value, terminated, truncated, gamma, gae_lambda):
    # 超时保留末观测的 bootstrap，但任何 episode 边界都切断优势递推。
    bootstrap = (~terminated.bool()).to(value.dtype)
    continuation = (~(terminated.bool() | truncated.bool())).to(value.dtype)
    delta = reward + gamma * bootstrap * next_value - value
    advantage = torch.empty_like(value)
    carry = torch.zeros_like(value[0])
    for step in reversed(range(len(value))):
        carry = delta[step] + gamma * gae_lambda * continuation[step] * carry
        advantage[step] = carry
    return advantage, advantage + value
