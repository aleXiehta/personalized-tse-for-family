import itertools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tasnet import choose_layer_norm
from models.film import FiLM1d

EPS = 1e-12

class WaveSplitBase(nn.Module):
    def __init__(self, speaker_stack: nn.Module, separation_stack: nn.Module, n_sources=2, n_training_sources=10, spk_criterion=None):
        super().__init__()

        assert spk_criterion is not None, "Specify spk_criterion."

        self.speaker_stack = speaker_stack
        self.separation_stack = separation_stack

        all_spk_idx = torch.arange(n_training_sources).long()
        self.all_spk_idx = nn.Parameter(all_spk_idx, requires_grad=False)

        self.spk_criterion = spk_criterion

        self.n_sources, self.n_training_sources = n_sources, n_training_sources

    def forward(self, mixture, spk_idx=None, sorted_idx=None, return_all_layers=False, return_spk_vector=False, stack_dim=1):
        """
        Only supports training time
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
        Returns:
            estimated_sources: (batch_size, num_blocks * num_layers, 1, T) if stack_dim=1.
            sorted_spk_vector: (batch_size, n_sources, latent_dim, T)
        """
        if self.training:
            if sorted_idx is None:
                if return_all_layers or return_spk_vector:
                    raise ValueError("Set return_all_layers=False, return_spk_vector=False.")
                
                sorted_idx = self.solve_permutation(mixture, spk_idx=spk_idx) # (batch_size, T, n_sources)

                return sorted_idx
        
            # If sorted_idx is given.
            estimated_sources, sorted_spk_vector = self.extract_latent(mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim)
        else:
            if spk_idx is not None:
                if sorted_idx is None:
                    if return_all_layers or return_spk_vector:
                        raise ValueError("Set return_all_layers=False, return_spk_vector=False.")
                    
                    sorted_idx = self.solve_permutation(mixture, spk_idx=spk_idx) # (batch_size, T, n_sources)

                    return sorted_idx

                # If sorted_idx is given.
                estimated_sources, sorted_spk_vector = self.extract_latent(mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim)
            else:
                spk_vector = self.speaker_stack(mixture) # (batch_size, n_sources, latent_dim, T)
                sorted_spk_vector = self.apply_kmeans(spk_vector, feature_last=False) # (batch_size, n_sources, latent_dim, T)
                spk_centroids = sorted_spk_vector.mean(dim=-1) # (batch_size, n_sources, latent_dim)
                estimated_sources = self.separation_stack(mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim)
        
        if return_spk_vector:
            return estimated_sources, sorted_spk_vector
            
        return estimated_sources

    def extract_latent(self, mixture, sorted_idx, return_all_layers=False, stack_dim=1):
        """
        Only supports training time
        Args:
            mixture: (batch_size, 1, T)
            sorted_idx: (batch_size, T, n_sources)
        Returns:
            estimated_sources:
                (batch_size, num_blocks * num_layers, n_sources, T) if return_all_layers=True and stack_dim=1.
                (batch_size, n_sources, T) if return_all_layers=False.
            sorted_spk_vector: Speaker vector with shape of (batch_size, n_sources, latent_dim, T), sorted by speaker loss.
        """
        spk_vector = self.speaker_stack(mixture) # (batch_size, n_sources, latent_dim, T)

        spk_vector = spk_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dim)

        # Use oracle sorted_idx during training. You can use oracle sorted_idx during evaluation if speakers in validation set are equal to training one.
        mask = torch.eye(self.n_sources)[sorted_idx] # (batch_size, T, n_sources, n_sources)
        mask = mask.to(spk_vector.device)
        sorted_spk_vector = torch.sum(mask.unsqueeze(dim=4) * spk_vector.unsqueeze(dim=3), dim=2) # (batch_size, T, n_sources, latent_dim)
        sorted_spk_vector = sorted_spk_vector.permute(0, 2, 3, 1).contiguous() # (batch_size, n_sources, latent_dim, T)
        spk_centroids = sorted_spk_vector.mean(dim=3) # (batch_size, n_sources, latent_dim)
        
        estimated_sources = self.separation_stack(mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim)

        return estimated_sources, sorted_spk_vector

    def solve_permutation(self, mixture, spk_idx):
        """
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
        Returns:
            sorted_idx: (batch_size, T, n_sources)
        """
        raise NotImplementedError("Implement solve_permutation.")

    def compute_pit_speaker_loss(self, spk_vector, spk_embedding, all_spk_embedding, feature_last=True, batch_mean=True):
        """
        Args:
            spk_vector: (batch_size, T, n_sources, latent_dim)
            spk_embedding: (batch_size, n_sources, latent_dim)
            all_spk_embedding: (batch_size, n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T) or (T,)
            pattern: (batch_size, T, P)
        """
        assert feature_last, "feature_last should be True."

        patterns = list(itertools.permutations(range(self.n_sources)))
        patterns = torch.Tensor(patterns).long()
        
        P = len(patterns)
        possible_loss = []
        
        for idx in range(P):
            pattern = patterns[idx]
            loss = self.spk_criterion(spk_vector[:, :, pattern], spk_embedding, all_spk_embedding, feature_last=feature_last, batch_mean=False, time_mean=False) # (batch_size, T)
            possible_loss.append(loss)
        
        possible_loss = torch.stack(possible_loss, dim=2) # (batch_size, T, P)
        loss, indices = torch.min(possible_loss, dim=2) # loss (batch_size, T), indices (batch_size, T)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss, patterns[indices]
    
    def apply_kmeans(self, spk_vector, feature_last=False, iter_clustering=100):
        """
        Args:
            spk_vector: (batch_size, n_sources, latent_dim, T) or (batch_size, T, n_sources, latent_dim)
        Returns:
            spk_vector: (batch_size, n_clusters, latent_dim, T), where n_clusters = n_sources
        """
        if not feature_last:
            spk_vector = spk_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dim)

        n_sources = self.n_sources

        for idx in range(iter_clustering):
            centroids = spk_vector.mean(dim=1, keepdim=True) # (batch_size, 1, n_clusters, latent_dim)
            distance = torch.norm(spk_vector.unsqueeze(dim=3) - centroids.unsqueeze(dim=2), dim=4) # (batch_size, T, n_sources, n_clusters)
            cluster_idx = torch.argmin(distance, dim=3) # (batch_size, T, n_sources)
            mask = torch.eye(n_sources)[cluster_idx] # (batch_size, T, n_sources, n_clusters)
            mask = mask.to(spk_vector.device)
            spk_vector = torch.sum(mask.unsqueeze(dim=4) * spk_vector.unsqueeze(dim=3), dim=2) # (batch_size, T, n_clusters, latent_dim)

        if not feature_last:    
            spk_vector = spk_vector.permute(0, 2, 3, 1).contiguous() # (batch_size, n_clusters, latent_dim, T)
        
        return spk_vector

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        # Ignore parameters of spk_criterion
        for p in self.spk_criterion.parameters():
            if p.requires_grad:
                _num_parameters -= p.numel()

        return _num_parameters

    def get_config(self):
        config = {
            'base': {
                'n_sources': self.n_sources,
                'n_training_sources': self.n_training_sources
            },
            'spk_stack': self.speaker_stack.get_config(),
            'sep_stack': self.separation_stack.get_config()
        }

        return config

    @classmethod
    def build_model(cls, model_path, spk_stack_cls, sep_stack_cls, spk_criterion, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        speaker_stack = spk_stack_cls(
            **config['spk_stack']
        )

        separation_stack = sep_stack_cls(
            **config['sep_stack']
        )

        n_sources, n_training_sources = config['base']['n_sources'], config['base']['n_training_sources']

        model = cls(speaker_stack, separation_stack, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

class WaveSplit(WaveSplitBase):
    def __init__(self, speaker_stack: nn.Module, separation_stack: nn.Module, latent_dim: int, n_sources=2, n_training_sources=10, spk_criterion=None, eps=EPS):
        super().__init__(speaker_stack, separation_stack, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)

        warnings.warn("Implementation of WaveSplit has not finished.", UserWarning)
    
        self.embedding = nn.Embedding(n_training_sources, latent_dim)

        self.eps = eps

    def forward(self, mixture, spk_idx=None, sorted_idx=None, return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False, stack_dim=1):
        """
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
        Returns:
            sorted_idx: (batch_size, T, n_sources) if sorted_idx is None.
            estimated_sources:
                (batch_size, num_blocks * num_layers, n_sources, T) if return_all_layers=True and stack_dim=1.
                (batch_size, n_sources, T) if return_all_layers=False.
            sorted_spk_vector: Speaker vector with shape of (batch_size, n_sources, latent_dim, T), sorted by speaker loss. If return_spk_vector=True, sorted_spk_vector is returned.
            spk_embedding: (batch_size, n_sources, latent_dim)
            all_spk_embedding: (n_training_sources, latent_dim)
        """
        eps = self.eps

        if self.training:
            if sorted_idx is None:
                if return_all_layers or return_spk_vector or return_spk_embedding or return_all_spk_embedding:
                    raise ValueError("Set return_all_layers=False, return_spk_vector=False, return_spk_embedding=False, return_all_spk_embedding=False.")
                
                sorted_idx = self.solve_permutation(mixture, spk_idx=spk_idx) # (batch_size, T, n_sources)

                return sorted_idx
        
            estimated_sources, sorted_spk_vector = self.extract_latent(mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim)
            
            if return_spk_embedding:
                spk_embedding = self.embedding(spk_idx) # (batch_size, n_sources, latent_dim)
                spk_embedding = spk_embedding / (torch.linalg.vector_norm(spk_embedding, dim=2, keepdim=True) + eps)
        else:
            if spk_idx is not None:
                if sorted_idx is None:
                    if return_all_layers or return_spk_vector:
                        raise ValueError("Set return_all_layers=False, return_spk_vector=False.")
                    
                    sorted_idx = self.solve_permutation(mixture, spk_idx=spk_idx) # (batch_size, T, n_sources)

                    return sorted_idx

                # If sorted_idx is given.
                estimated_sources, sorted_spk_vector = self.extract_latent(mixture, sorted_idx, return_all_layers=return_all_layers, stack_dim=stack_dim)

                if return_spk_embedding:
                    spk_embedding = self.embedding(spk_idx) # (batch_size, n_sources, latent_dim)
                    spk_embedding = spk_embedding / (torch.linalg.vector_norm(spk_embedding, dim=2, keepdim=True) + eps)
            else:
                spk_vector = self.speaker_stack(mixture) # (batch_size, n_sources, latent_dim, T)
                sorted_spk_vector = self.apply_kmeans(spk_vector, feature_last=False) # (batch_size, n_sources, latent_dim, T)
                spk_centroids = sorted_spk_vector.mean(dim=-1) # (batch_size, n_sources, latent_dim), works as centroids
                spk_embedding = spk_centroids / (torch.linalg.vector_norm(spk_centroids, dim=2, keepdim=True) + eps)
                estimated_sources = self.separation_stack(mixture, spk_centroids, return_all=return_all_layers, stack_dim=stack_dim)
        
        output = []
        output.append(estimated_sources)

        if return_spk_vector:
            output.append(sorted_spk_vector)
        
        if return_spk_embedding:
            output.append(spk_embedding)
        
        if return_all_spk_embedding:    
            all_spk_embedding = self.embedding(self.all_spk_idx) # (n_training_sources, latent_dim)
            all_spk_embedding = all_spk_embedding / (torch.linalg.vector_norm(all_spk_embedding, dim=1, keepdim=True) + eps)
            output.append(all_spk_embedding)
        
        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)

        return output

    def solve_permutation(self, mixture, spk_idx):
        """
        Args:
            mixture: (batch_size, 1, T)
            spk_idx: (batch_size, n_sources)
        Returns:
            sorted_idx: (batch_size, T, n_sources)
        """
        eps = self.eps

        spk_vector = self.speaker_stack(mixture) # (batch_size, n_sources, latent_dim, T)
        spk_vector = spk_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dim)
        
        spk_embedding = self.embedding(spk_idx) # (batch_size, n_sources, latent_dim)
        all_spk_embedding = self.embedding(self.all_spk_idx) # (n_training_sources, latent_dim)

        spk_embedding = spk_embedding / (torch.linalg.vector_norm(spk_embedding, dim=2, keepdim=True) + eps)
        all_spk_embedding = all_spk_embedding / (torch.linalg.vector_norm(all_spk_embedding, dim=1, keepdim=True) + eps)

        _, sorted_idx = self.compute_pit_speaker_loss(spk_vector, spk_embedding, all_spk_embedding, feature_last=True, batch_mean=False) # (batch_size, T, n_sources)
    
        return sorted_idx

class SpeakerStack(nn.Module):
    def __init__(self, in_channels, latent_dim=512, kernel_size=3, num_layers=14, dilated=True, separable=True, causal=False, nonlinear=None, norm=True, n_sources=2, eps=EPS):
        super().__init__()
        
        self.num_layers = num_layers
        self.n_sources = n_sources

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.nonlinear, self.norm = nonlinear, norm

        self.eps = eps
        
        net = []
        
        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2

            if idx == 0:
                _in_channels = in_channels
                _out_channels = latent_dim
            elif idx == num_layers - 1:
                _in_channels = latent_dim
                _out_channels = n_sources * latent_dim
            else:
                _in_channels = latent_dim
                _out_channels = latent_dim
            
            block = ResidualBlock1d(
                _in_channels, _out_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                separable=separable, causal=causal, nonlinear=nonlinear, norm=norm,
                eps=eps
            )
            net.append(block)
        
        self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input: (batch_size, 1, T)
        Returns:
            output: (batch_size, n_sources, latent_dim, T)
        """
        n_sources = self.n_sources
        eps = self.eps
        x = input
        
        for idx in range(self.num_layers):
            x = self.net[idx](x)
        
        batch_size, _, T = x.size()
        output = x.view(batch_size, n_sources, -1, T)
        output = output / (torch.linalg.vector_norm(output, dim=2, keepdim=True) + eps)

        return output

    def get_config(self):
        config = {}

        config['in_channels'] = self.in_channels
        config['latent_dim'] = self.latent_dim
        config['kernel_size'] = self.kernel_size
        config['num_layers'] = self.num_layers

        config['dilated'], config['separable'], config['causal'] = self.dilated, self.separable, self.causal
        config['nonlinear'], config['norm'] = self.nonlinear, self.norm

        config['n_sources'] = self.n_sources

        config['eps'] = self.eps

        return config

    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels, latent_dim = config['in_channels'], config['latent_dim']
        kernel_size = config['kernel_size']
        num_layers = config['num_layers']
        dilated, separable, causal = config['dilated'], config['separable'], config['causal']
        nonlinear, norm = config['nonlinear'], config['norm']
        n_sources = config['n_sources']
        eps = config['eps']

        model = cls(
            in_channels, latent_dim, kernel_size=kernel_size,
            num_layers=num_layers,
            dilated=dilated, separable=separable, causal=causal,
            nonlinear=nonlinear, norm=norm, n_sources=n_sources,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

class SeparationStack(nn.Module):
    def __init__(self, in_channels, latent_dim=512, kernel_size_in=4, kernel_size=3, num_blocks=4, num_layers=10, dilated=True, separable=True, causal=False, nonlinear=None, norm=True, n_sources=2, eps=EPS):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kernel_size_in, self.kernel_size = kernel_size_in, kernel_size
        self.num_blocks, self.num_layers = num_blocks, num_layers

        self.dilated, self.separable, self.causal = dilated, separable, causal
        self.nonlinear, self.norm = nonlinear, norm

        self.n_sources = n_sources

        self.eps = eps

        self.conv1d = nn.Conv1d(in_channels, latent_dim, kernel_size=kernel_size_in, stride=1)
        
        net = []
        fc_weights, fc_biases = [], []

        for block_idx in range(num_blocks):
            subnet = []
            sub_fc_weights, sub_fc_biases = [], []

            for layer_idx in range(num_layers):
                if dilated:
                    dilation = 2**layer_idx
                    stride = 1
                else:
                    dilation = 1
                    stride = 2

                if block_idx == num_blocks - 1 and layer_idx == num_layers - 1:
                    dual_head = False
                else:
                    dual_head = True
                
                block = FiLMResidualBlock1d(
                    latent_dim, latent_dim, skip_channels=n_sources,
                    kernel_size=kernel_size, stride=stride, dilation=dilation,
                    separable=separable, causal=causal, nonlinear=nonlinear, norm=norm,
                    dual_head=dual_head,
                    eps=eps
                )
                subnet.append(block)

                # For FiLM
                weights = MultiSourceProjection1d(latent_dim, latent_dim, n_sources=n_sources)
                biases = MultiSourceProjection1d(latent_dim, latent_dim, n_sources=n_sources)
                sub_fc_weights.append(weights)
                sub_fc_biases.append(biases)
            
            subnet = nn.Sequential(*subnet)
            net.append(subnet)

            sub_fc_weights, sub_fc_biases = nn.ModuleList(sub_fc_weights), nn.ModuleList(sub_fc_biases)
            fc_weights.append(sub_fc_weights)
            fc_biases.append(sub_fc_biases)
        
        self.fc_weights = nn.ModuleList(fc_weights)
        self.fc_biases = nn.ModuleList(fc_biases)

        self.net = nn.Sequential(*net)

    def forward(self, input, spk_centroids, return_all=False, stack_dim=1):
        """
        Args:
            input (batch_size, in_channels, T)
            spk_centroids (batch_size, n_sources, latent_dim)
        Returns:
            output:
                (batch_size, num_blocks*num_layers, n_sources, T) if return_all
                (batch_size, n_sources, T) otherwise
        """
        padding = self.kernel_size_in - 1
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.conv1d(x)

        skip_connection = []
        
        for block_idx in range(self.num_blocks):
            fc_weights_block = self.fc_weights[block_idx]
            fc_biases_block = self.fc_biases[block_idx]
            net_block = self.net[block_idx]
            for layer_idx in range(self.num_layers):
                gamma = fc_weights_block[layer_idx](spk_centroids)
                beta = fc_biases_block[layer_idx](spk_centroids)
                x, skip = net_block[layer_idx](x, gamma, beta)
                skip_connection.append(skip)
        
        if return_all:
            output = torch.stack(skip_connection, dim=stack_dim)
        else:
            output = skip_connection[-1]

        return output
    
    def get_config(self):
        config = {}

        config['in_channels'] = self.in_channels
        config['latent_dim'] = self.latent_dim
        config['kernel_size_in'], config['kernel_size'] = self.kernel_size_in, self.kernel_size
        config['num_blocks'], config['num_layers'] = self.num_blocks, self.num_layers

        config['dilated'], config['separable'], config['causal'] = self.dilated, self.separable, self.causal
        config['nonlinear'], config['norm'] = self.nonlinear, self.norm

        config['n_sources'] = self.n_sources
        config['eps'] = self.eps

        return config
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=True):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        in_channels = config['in_channels']
        latent_dim = config['latent_dim']
        kernel_size_in, kernel_size = config['kernel_size_in'], config['kernel_size']

        num_blocks, num_layers = config['num_blocks'], config['num_layers']

        dilated, separable, causal = config['dilated'], config['separable'], config['causal']
        nonlinear, norm = config['nonlinear'], config['norm']

        n_sources = config['n_sources']

        eps = config['eps']

        model = cls(
            in_channels, latent_dim, kernel_size_in=kernel_size_in, kernel_size=kernel_size,
            num_blocks=num_blocks, num_layers=num_layers,
            dilated=dilated, separable=separable, causal=causal,
            nonlinear=nonlinear, norm=norm, n_sources=n_sources,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])

        return model

class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels=512, kernel_size=3, stride=1, dilation=1, separable=True, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm
        
        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, eps=eps)
        else:            
            self.conv1d = ConvBlock1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, eps=eps)

    def forward(self, input):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        separable, causal = self.separable, self.causal
        
        _, _, T_original = input.size()
        
        residual = input
        
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        
        if separable:
            output = self.separable_conv1d(x)
        else:
            output = self.conv1d(x)
        
        if input.size(1) == output.size(1):
            output = output + residual
            
        return output

class FiLMResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels=512, skip_channels=2, kernel_size=3, stride=1, dilation=1, separable=True, causal=False, nonlinear=None, norm=True, dual_head=False, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.dual_head = dual_head
        self.norm = norm
        
        if separable:
            self.output_conv1d = FiLMDepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, eps=eps)
        else:
            self.output_conv1d = FiLMConvBlock1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, eps=eps)
        
        self.skip_conv1d = nn.Conv1d(out_channels, skip_channels, kernel_size=1, stride=1, dilation=1)

    def forward(self, input, gamma, beta):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        causal = self.causal
        dual_head = self.dual_head
        
        _, _, T_original = input.size()
        
        residual = input
        
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.output_conv1d(x, gamma, beta)
        x = x + residual
        skip = self.skip_conv1d(x)

        if dual_head:
            output = x
        else:
            output = None
        
        return output, skip

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels=512, kernel_size=3, stride=1, dilation=1, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.norm = norm
        self.eps = eps
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)

        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)
        
    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm
        
        x = self.conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)
        
        if norm:
            x = self.norm1d(x)
        
        output = x
        
        return output

class FiLMConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels=512, kernel_size=3, stride=1, dilation=1, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.norm = norm
        self.eps = eps
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        self.film1d = FiLM1d()

        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)
        
    def forward(self, input, gamma, beta):
        nonlinear, norm = self.nonlinear, self.norm
        
        x = self.conv1d(input)
        x = self.film1d(x, gamma, beta)

        if nonlinear:
            x = self.nonlinear1d(x)
        
        if norm:
            x = self.norm1d(x)
        
        output = x
        
        return output

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=512, kernel_size=3, stride=1, dilation=1, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.norm = norm
        self.eps = eps
        
        self.depthwise_conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)  
        
    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm
        
        x = self.depthwise_conv1d(input)
        x = self.pointwise_conv1d(x)

        if nonlinear:
            x = self.nonlinear1d(x)
        
        if norm:
            x = self.norm1d(x)
        
        output = x
        
        return output

class FiLMDepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=512, kernel_size=3, stride=1, dilation=1, causal=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.norm = norm
        self.eps = eps
        
        self.depthwise_conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.film1d = FiLM1d()

        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, out_channels, causal=causal, eps=eps)
        
    def forward(self, input, gamma, beta):
        nonlinear, norm = self.nonlinear, self.norm
        
        x = self.depthwise_conv1d(input)
        x = self.pointwise_conv1d(x)
        x = self.film1d(x, gamma, beta)

        if nonlinear:
            x = self.nonlinear1d(x)
        
        if norm:
            x = self.norm1d(x)
        
        output = x
        
        return output

class MultiSourceProjection1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_sources, channel_last=True):
        super().__init__()
        
        assert channel_last, "channel_last should be True."
        self.linear = nn.Linear(n_sources * in_channels, out_channels)
    
    def forward(self, input):
        batch_size, n_sources, in_channels = input.size()

        x = input.view(batch_size, n_sources*in_channels)
        output = self.linear(x)

        return output

class _SpeakerDistance(nn.Module):
    def __init__(self, n_sources):
        super().__init__()

        self.mask = nn.Parameter(1 - torch.eye(n_sources), requires_grad=False)

        zero_dim_size = ()
        self.scale, self.bias = nn.Parameter(torch.empty(zero_dim_size), requires_grad=True), nn.Parameter(torch.empty(zero_dim_size), requires_grad=True)

        self._reset_parameters()
    
    def _reset_parameters(self):
        self.scale.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, spk_vector, spk_embedding, _, feature_last=True, batch_mean=True, time_mean=True):
        """
        Args:
            spk_vector:
                (batch_size, T, n_sources, latent_dim) if feature_last
                (batch_size, n_sources, latent_dim, T) otherwise
            spk_embedding: (batch_size, n_sources, latent_dim)
            _: All speaker embedding (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T) or (T,)
        """
        if not feature_last:
            spk_vector = spk_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dims)

        loss_euclid = self.compute_euclid_distance(spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1) # (batch_size, T, n_sources)

        distance_table = self.compute_euclid_distance(spk_vector.unsqueeze(dim=3), spk_vector.unsqueeze(dim=2), dim=-1) # (batch_size, T, n_sources, n_sources)
        loss_hinge = F.relu(1 - distance_table) # (batch_size, T, n_sources, n_sources)
        loss_hinge = torch.sum(self.mask * loss_hinge, dim=2) # (batch_size, T, n_sources)

        loss = loss_euclid + loss_hinge
        loss = loss.mean(dim=-1)

        if time_mean:
            loss = loss.mean(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        return loss
    
    def compute_euclid_distance(self, input, target, dim=-1, keepdim=False, scale=None, bias=None):
        distance = torch.sum((input - target)**2, dim=dim, keepdim=keepdim)

        if scale is not None or bias is not None:
            distance = torch.abs(self.scale) * distance + self.bias
        
        return distance

class _SpeakerLoss(nn.Module):
    def __init__(self, n_sources):
        super().__init__()

        self.mask = nn.Parameter(1 - torch.eye(n_sources), requires_grad=False)

        zero_dim_size = ()
        self.scale, self.bias = nn.Parameter(torch.empty(zero_dim_size), requires_grad=True), nn.Parameter(torch.empty(zero_dim_size), requires_grad=True)

        self._reset_parameters()
    
    def _reset_parameters(self):
        self.scale.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, speaker_vector, speaker_embedding, all_speaker_embedding, feature_last=True, batch_mean=True, time_mean=True):
        """
        Args:
            speaker_vector:
                (batch_size, T, n_sources, latent_dims) if feature_last=True
                (batch_size, n_sources, latent_dims, T) otherwise
            speaker_embedding: (batch_size, n_sources, latent_dim)
            all_speaker_embedding: (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size,) or ()
        """
        if not feature_last:
            speaker_vector = speaker_vector.permute(0, 3, 1, 2).contiguous() # (batch_size, T, n_sources, latent_dims)
        
        loss = self.compute_speaker_loss(speaker_vector, speaker_embedding, all_speaker_embedding, scale=self.scale, bias=self.bias, feature_last=True, batch_mean=False) # (batch_size, T, n_sources)
        loss = loss.mean(dim=-1)

        if time_mean:
            loss = loss.mean(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_speaker_loss(self, spk_vector, spk_embedding, all_spk_embedding, scale=None, bias=None, feature_last=True, batch_mean=True):
        """
        Args:
            spk_vector: (batch_size, T, n_sources, latent_dim)
            spk_embedding: (batch_size, n_sources, latent_dim)
            all_spk_embedding: (n_training_sources, latent_dim)
        Returns:
            loss: (batch_size, T, n_sources) or (T, n_sources)
        """
        assert feature_last, "feature_last should be True."

        loss_distance = self.compute_speaker_distance(spk_vector, spk_embedding, feature_last=feature_last, batch_mean=False) # (batch_size, T, n_sources)

        rescaled_distance = self.compute_euclid_distance(spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources)
        rescaled_all_distance = self.compute_euclid_distance(spk_vector.unsqueeze(dim=3), all_spk_embedding, dim=-1, scale=scale, bias=bias) # (batch_size, T, n_sources, n_training_sources)

        loss_local = self.compute_local_classification(rescaled_distance, batch_mean=False) # (batch_size, T, n_sources)
        loss_global = self.compute_global_classification(rescaled_distance, rescaled_all_distance, batch_mean=False) # (batch_size, T, n_sources)
        loss = loss_distance + loss_local + loss_global

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_speaker_distance(self, spk_vector, spk_embedding, feature_last=True, batch_mean=True):
        """
        Args:
            spk_vector: (batch_size, T, n_sources, latent_dim)
            spk_embedding: (batch_size, n_sources, latent_dim)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        assert feature_last, "feature_last should be True."

        loss_distance = self.compute_euclid_distance(spk_vector, spk_embedding.unsqueeze(dim=1), dim=-1) # (batch_size, T, n_sources)

        distance_table = self.compute_euclid_distance(spk_vector.unsqueeze(dim=3), spk_vector.unsqueeze(dim=2), dim=-1) # (batch_size, T, n_sources, n_sources)
        loss_hinge = F.relu(1 - distance_table) # (batch_size, T, n_sources, n_sources)
        loss_hinge = torch.sum(self.mask * loss_hinge, dim=2) # (batch_size, T, n_sources)

        loss = loss_distance + loss_hinge

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
    
    def compute_local_classification(self, distance, batch_mean=True):
        """
        Args:
            distance: (batch_size, T, n_sources)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        loss = distance + torch.logsumexp(- distance, dim=2, keepdim=True) # (batch_size, T, n_sources)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    def compute_global_classification(self, distance, all_distance, batch_mean=True):
        """
        Args:
            distance: (batch_size, T, n_sources)
            all_distance: (batch_size, T, n_sources, n_training_sources)
        Returns:
            loss: (batch_size, T, n_sources)
        """
        loss = distance + torch.logsumexp(- all_distance, dim=3) # (batch_size, T, n_sources)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    def compute_euclid_distance(self, input, target, dim=-1, keepdim=False, scale=None, bias=None):
        distance = torch.sum((input - target)**2, dim=dim, keepdim=keepdim)

        if scale is not None or bias is not None:
            distance = torch.abs(self.scale) * distance + self.bias
        
        return distance

def _test_wavesplit():
    batch_size = 4
    n_training_sources, n_sources = 10, 2
    in_channels = 1
    latent_dim = 8
    T = 1024

    input = torch.randn(batch_size, in_channels, T)
    target = torch.randn(batch_size, n_sources, T)
    spk_idx = torch.randint(0, n_training_sources, (batch_size, n_sources))

    spk_criterion = _SpeakerLoss(n_sources=n_sources)

    speaker_stack = SpeakerStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    separation_stack = SeparationStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    model = WaveSplit(speaker_stack, separation_stack, latent_dim, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)

    with torch.no_grad():
        sorted_idx = model(input, spk_idx=spk_idx, return_all_layers=False, return_spk_vector=False)
    
    model.zero_grad()

    estimated_sources, sorted_spk_vector = model(input, spk_idx=spk_idx, sorted_idx=sorted_idx, return_all_layers=True, return_spk_vector=True, return_spk_embedding=False, return_all_spk_embedding=False)

    loss = - sisdr(estimated_sources, target.unsqueeze(dim=1))
    loss = loss.mean()

    print(model)
    print(input.size(), estimated_sources.size(), sorted_idx.size(), sorted_spk_vector.size())

def _test_wavesplit_spk_distance():
    batch_size = 4
    n_training_sources, n_sources = 10, 2
    in_channels = 1
    latent_dim = 8
    T = 1024
    
    input = torch.randn(batch_size, in_channels, T)
    target = torch.randn(batch_size, n_sources, T)
    spk_idx = torch.randint(0, n_training_sources, (batch_size, n_sources))

    spk_criterion = _SpeakerDistance(n_sources=n_sources)

    speaker_stack = SpeakerStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    separation_stack = SeparationStack(in_channels, latent_dim, num_layers=5, separable=False, n_sources=n_sources)
    model = WaveSplit(speaker_stack, separation_stack, latent_dim, n_sources=n_sources, n_training_sources=n_training_sources, spk_criterion=spk_criterion)

    with torch.no_grad():
        sorted_idx = model(input, spk_idx=spk_idx, return_all_layers=False, return_spk_vector=False)
    
    model.zero_grad()

    estimated_sources, sorted_spk_vector = model(input, spk_idx=spk_idx, sorted_idx=sorted_idx, return_all_layers=True, return_spk_vector=True, return_spk_embedding=False, return_all_spk_embedding=False)

    loss = - sisdr(estimated_sources, target.unsqueeze(dim=1))
    loss = loss.mean()

    print(model)
    print(model.num_parameters)
    print(input.size(), estimated_sources.size(), sorted_idx.size(), sorted_spk_vector.size())
    print()

    model.eval()
    estimated_sources, sorted_spk_vector = model(input, return_all_layers=False, return_spk_vector=True)

    print(input.size(), estimated_sources.size(), sorted_idx.size(), sorted_spk_vector.size())
    print()

if __name__ == '__main__':
    import torch

    from criterion.sdr import sisdr

    torch.manual_seed(111)

    print("="*10, "WaveSplit", "="*10)
    _test_wavesplit()

    print("="*10, "WaveSplitOracle", "="*10)
    _test_wavesplit_spk_distance()