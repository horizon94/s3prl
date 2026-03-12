import os
from .expert import FlmaudioUpstream as _UpstreamExpert


def flmaudio(
    ckpt=None,
    model_config=None,
    mimi_ckpt=None,
    teleflm_ckpt=None,
    teleflm_config=None,
    num_quantizers=8,
    hidden_upsample_factor=None,
    slow_factor=None,
    **kwargs
):
    """
    Flmaudio upstream model
    
    参数可以通过以下方式传递：
    1. 通过命令行参数 -k (upstream_ckpt) 和 -g (upstream_model_config)
    2. 通过环境变量
    3. 通过直接调用时的命名参数
    
    Args:
        ckpt: TeleFLM checkpoint 路径（对应 -k 参数）
        model_config: TeleFLM config 路径（对应 -g 参数）
        mimi_ckpt: MimiModel checkpoint 路径（可通过环境变量 FLMAUDIO_MIMI_CKPT 设置）
        teleflm_ckpt: TeleFLM checkpoint 路径（如果提供，会覆盖 ckpt）
        teleflm_config: TeleFLM config 路径（如果提供，会覆盖 model_config）
        num_quantizers: MimiModel 编码时使用的 quantizer 数量（默认: 8）
    
    命令行使用示例:
        # 方式1: 使用环境变量设置 mimi_ckpt
        export FLMAUDIO_MIMI_CKPT=/path/to/mimi/model
        python run_downstream.py -u flmaudio -k /path/to/teleflm/ckpt -g /path/to/teleflm/config.json ...
        
        # 方式2: 使用默认的 HuggingFace mimi 模型
        python run_downstream.py -u flmaudio -k /path/to/teleflm/ckpt -g /path/to/teleflm/config.json ...
    """
    # 优先级: 直接参数 > ckpt/model_config > 环境变量 > 默认值
    
    # 处理 teleflm_ckpt
    if teleflm_ckpt is None:
        teleflm_ckpt = ckpt
    
    # 处理 teleflm_config
    if teleflm_config is None:
        teleflm_config = model_config
    
    # 处理 mimi_ckpt（从环境变量读取或使用默认值）
    if mimi_ckpt is None:
        mimi_ckpt = os.environ.get("FLMAUDIO_MIMI_CKPT", None)
        # 如果环境变量也没有，使用默认的 HuggingFace 模型名称
        if mimi_ckpt is None:
            mimi_ckpt = "kyutai/mimi"  # 默认值

    if hidden_upsample_factor is None:
        hidden_upsample_factor = int(os.environ.get("FLMAUDIO_HIDDEN_UPSAMPLE", "1"))
    if slow_factor is None:
        slow_factor = float(os.environ.get("FLMAUDIO_SLOW_FACTOR", "1.0"))
    
    return _UpstreamExpert(
        mimi_ckpt=mimi_ckpt,
        teleflm_ckpt=teleflm_ckpt,
        teleflm_config=teleflm_config,
        num_quantizers=num_quantizers,
        hidden_upsample_factor=hidden_upsample_factor,
        slow_factor=slow_factor,
        **kwargs
    )


def flmaudio_local(ckpt, model_config=None, **kwargs):
    """
    本地 checkpoint 版本（与 flmaudio 相同，但明确表示使用本地文件）
    """
    return flmaudio(
        ckpt=ckpt,
        model_config=model_config,
        **kwargs
    )


def flmaudio_custom(*args, **kwargs):
    """自定义配置的 flmaudio"""
    return flmaudio(*args, **kwargs)
