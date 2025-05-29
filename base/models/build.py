# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Rao Fu, RainbowSecret
# --------------------------------------------------------

from .hrnet import HighResolutionNet


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "hrnet":
        model = HighResolutionNet(
            config.MODEL.HRNET, num_classes=config.MODEL.NUM_CLASSES
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    print(model)
    return model