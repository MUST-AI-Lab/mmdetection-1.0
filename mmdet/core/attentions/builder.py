from mmcv.utils import Registry, build_from_cfg

ATTENTIONS = Registry('attentions')

def build_attention(cfg, **default_args):
    return build_from_cfg(cfg, ATTENTIONS, default_args)
