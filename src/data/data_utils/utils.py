from torch import Tensor
import torch

def normalize(context):
    context = context.replace('<define>', '[DEF]')
    context = context.replace('</define>', '[/DEF]')
    return context

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float())
    pooler_output = torch.sum(
        last_hidden_states * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooler_output
    # last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

