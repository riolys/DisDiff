from typing import List
import torch
from torch.nn import functional as F
from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention

class Attmap():
    def __init__(self, prompt, tokenizer) -> None:
        self.prompt = prompt
        self.tokenizer = tokenizer
    def _compute_max_attention_per_index(self,
                                        attention_maps: torch.Tensor,
                                        indices_to_alter: List[int],
                                        smooth_attentions: bool = True,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        normalize_eot: bool = True) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]
        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                indices_to_alter: List[int],
                                                attention_res: int = 16,
                                                smooth_attentions: bool = False,
                                                sigma: float = 0.5,
                                                kernel_size: int = 3,
                                                normalize_eot: bool = False):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        return max_attention_per_index
    
    def _get_attention_map(self, attention_store: AttentionStore,
                                                indices_to_alter: List[int],
                                                attention_res: int = 16):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        map = attention_maps[:, :, indices_to_alter]
        return map
    
    @staticmethod
    def _compute_AE_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss
        
    @staticmethod
    def _compute_energy_loss(map, token2ignore):
        #map -- [16, 16, len(prompt)]
        map = F.softmax(map * 100, dim=-1)
        token_attention = torch.sum(torch.square(map[:, :, token2ignore]))
        return (1 - token_attention/torch.sum(torch.square(map))) ** 2
    
    @staticmethod
    def _compute_energy_loss2(map, token2ignore):
        #map -- [16, 16, len(prompt)]
        max_map = torch.sum(map, axis=(0,1))
        token2ignore = torch.argmax(max_map)
        map = F.softmax(map * 100, dim=-1)
        token_attention = torch.sum(torch.square(map[:, :, token2ignore]))
        return (1 - token_attention/torch.sum(torch.square(map))) ** 2
    
    @staticmethod
    def _compute_l1_loss(map, token2ignore):
        return  -torch.mean(map[:, :, token2ignore])
    
    def forward(self, controller, loss_func, token2ignore: List[int]):
        prompt_length = len(self.prompt.split())
        indices_to_alter = list(range(1, prompt_length + 1))
        token2ignore = [i - 1 for i in token2ignore] 
        map = self._get_attention_map(attention_store=controller, indices_to_alter=indices_to_alter)
        if loss_func == "l1":
            return  self._compute_l1_loss(map, token2ignore)
        elif loss_func == "energy":
            return self._compute_energy_loss(map, token2ignore)
        elif loss_func == "energy2":
            return self._compute_energy_loss2(map, token2ignore)
        else:
            raise ValueError("loss function should be l1 or energy")
