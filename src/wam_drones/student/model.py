"""MobileNetV2-0.35 student with a normalised 64-D embedding head."""

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2  # type: ignore[import-untyped]


class StudentEmbeddingModel(nn.Module):
    """MCU-oriented convolutional image encoder; no class softmax head."""

    def __init__(self, embedding_dimension: int = 64) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=None, width_mult=0.35)
        input_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Linear(input_features, embedding_dimension)
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        """Return unit-length embeddings for NCHW 160x160 RGB tensors."""
        return F.normalize(self.backbone(images), dim=-1)
