from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from .wide_resnet import WideResNet28, WideResNet34
from .senet import SENet18

def get_model(model_name: str, num_classes: int=10):
    match model_name:
        case "ResNet18":
            return ResNet18(num_classes=num_classes)
        case "ResNet34":
            return ResNet34(num_classes=num_classes)
        case "ResNet50":
            return ResNet50(num_classes=num_classes)
        case "ResNet101":
            return ResNet101(num_classes=num_classes)
        case "ResNet152":
            return ResNet152(num_classes=num_classes)
        case "PreActResNet18":
            return PreActResNet18(num_classes=num_classes)
        case "PreActResNet34":
            return PreActResNet34(num_classes=num_classes)
        case "PreActResNet50":
            return PreActResNet50(num_classes=num_classes)
        case "PreActResNet101":
            return PreActResNet101(num_classes=num_classes)
        case "PreActResNet152":
            return PreActResNet152(num_classes=num_classes)
        case "WideResNet28":
            return WideResNet28(num_classes=num_classes)
        case "WideResNet34":
            return WideResNet34(num_classes=num_classes)
        case "SENet18":
            return SENet18(num_classes=num_classes)
        case _:
            raise "Invalid Model!"