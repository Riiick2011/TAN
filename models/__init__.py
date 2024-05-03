from .tan.build import build_tan


def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):  # 该函数用于构建一个tan检测器，并根据是否处于训练中返回损失函数
    # build action detector
    if 'tan_' in args.version:
        model, criterion = build_tan(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            resume=resume
            )

    return model, criterion

