"""
network.py
----------
A compact residual CNN that evaluates a chess position.
Now upgraded to optionally act as a full AlphaZero dual-headed network.

Architecture:
    input (21, 8, 8) for V2, or (18, 8, 8) for V1.
      -> conv stem (3x3, channels) + BN + ReLU
      -> N residual blocks with Squeeze-and-Excitation
      -> value head: 1x1 conv -> FC(256) -> FC(1) -> tanh
      -> policy head (optional): 1x1 conv -> flatten -> 4672 logits

Output:
    If output_policy=False (default): returns scalar in (-1, 1).
    If output_policy=True: returns (policy_logits, value_scalar).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.board_encoder import NUM_PLANES, NUM_PLANES_V2
from engine.move_encoding import NUM_MOVE_PLANES, TOTAL_MOVES

CP_SCALE = 400.0


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for dynamic channel weighting."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        return F.relu(out)


class ChessValueNet(nn.Module):
    def __init__(self, channels: int = 128, num_res_blocks: int = 15, num_planes: int = NUM_PLANES_V2, output_policy: bool = False):
        super().__init__()
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.output_policy = output_policy

        self.stem = nn.Sequential(
            nn.Conv2d(num_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Value head
        self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Policy head (always initialized so weights can be loaded, but only used if output_policy=True)
        self.policy_conv = nn.Conv2d(channels, NUM_MOVE_PLANES, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(NUM_MOVE_PLANES)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        # Value path
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (batch,)

        if not self.output_policy:
            return v
        
        # Policy path
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)  # (batch, 4672)
        return p, v

    @torch.no_grad()
    def evaluate(self, board_tensor: torch.Tensor, device=None) -> float:
        """Legacy compatibility for value-only evaluation."""
        self.eval()
        if not torch.is_tensor(board_tensor):
            board_tensor = torch.from_numpy(board_tensor)
        board_tensor = board_tensor.unsqueeze(0).float()
        if device is not None:
            board_tensor = board_tensor.to(device)
        
        out = self.forward(board_tensor)
        if self.output_policy:
            _, v = out
            return v.item()
        return out.item()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path: str, device=None, output_policy: bool = False) -> ChessValueNet:
    """Helper to load a checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = ChessValueNet(
        channels=ckpt.get("channels", 128),
        num_res_blocks=ckpt.get("num_res_blocks", 15),
        num_planes=NUM_PLANES_V2 if output_policy else NUM_PLANES,
        output_policy=output_policy
    )
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.to(device)
    net.eval()
    return net

def save_model(net: ChessValueNet, path: str, extra_meta: dict = None):
    """Helper to save a checkpoint."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model_state_dict": net.state_dict(),
        "channels": net.channels,
        "num_res_blocks": net.num_res_blocks,
    }
    if extra_meta:
        state.update(extra_meta)
    torch.save(state, path)


if __name__ == "__main__":
    net = ChessValueNet(output_policy=True)
    dummy = torch.randn(4, NUM_PLANES_V2, 8, 8)
    p, v = net(dummy)
    print("Policy shape:", p.shape)
    print("Value shape:", v.shape)
    print("param count:", count_parameters(net))
