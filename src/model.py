import timm

def deepfake_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=2
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model