from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from .wide_resnet import WideResNet28, WideResNet34
from .senet import SENet18

def get_model(model_name: str):
    match model_name:
        case "ResNet18":
            return ResNet18()
        case "ResNet34":
            return ResNet34()
        case "ResNet50":
            return ResNet50()
        case "ResNet101":
            return ResNet101()
        case "ResNet152":
            return ResNet152()
        case "PreActResNet18":
            return PreActResNet18()
        case "PreActResNet34":
            return PreActResNet34()
        case "PreActResNet50":
            return PreActResNet50()
        case "PreActResNet101":
            return PreActResNet101()
        case "PreActResNet152":
            return PreActResNet152()
        case "WideResNet28":
            return WideResNet28()
        case "WideResNet34":
            return WideResNet34()
        case "SENet18":
            return SENet18()
        case _:
            raise "Invalid Model!"