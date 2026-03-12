import logging
import os
from pathlib import Path
from typing import List, Dict, Union, Optional

import torch
import torchaudio
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from s3prl.upstream.interfaces import UpstreamBase

from s3prl.upstream.flmaudio.audio_model.modeling_mimi import MimiModel
from s3prl.upstream.flmaudio.audio_model.modeling_teleflm import TeleFLMForCausalLM
from s3prl.upstream.flmaudio.audio_model.configuration_teleflm import TeleFLMConfig

logger = logging.getLogger(__name__)

# MimiModel 的采样率是 24000 Hz
MIMI_SAMPLE_RATE = 24000
# S3PRL 标准采样率是 16000 Hz
S3PRL_SAMPLE_RATE = 16000


class FlmaudioUpstream(UpstreamBase):
    """
    Flmaudio Upstream Model for S3PRL
    
    使用流程:
    1. MimiModel 将音频波形编码为离散 tokens (audio_codes)
    2. TeleFLMForCausalLM 处理 tokens 并输出 hidden states
    3. 通过 hook 机制提取各层特征
    """
    
    def __init__(
        self,
        mimi_ckpt: Optional[str] = None,
        teleflm_ckpt: Optional[str] = None,
        teleflm_config: Optional[str] = None,
        num_quantizers: int = 8,
        hidden_upsample_factor: int = 1,
        slow_factor: float = 1.0,
        **kwargs
    ):
        """
        Args:
            mimi_ckpt: MimiModel checkpoint 路径或 HuggingFace 模型名称
            teleflm_ckpt: TeleFLMForCausalLM checkpoint 路径
            teleflm_config: TeleFLMConfig 配置文件路径（如果 teleflm_ckpt 不包含 config）
            num_quantizers: MimiModel 编码时使用的 quantizer 数量
        """
        super().__init__(**kwargs)
        
        self.num_quantizers = num_quantizers
        self.hidden_upsample_factor = max(1, int(hidden_upsample_factor))
        self.slow_factor = float(slow_factor)
        if self.slow_factor < 1.0:
            raise ValueError(f"slow_factor must be >= 1.0, got {self.slow_factor}")
        self.acoustic_shift = 2
        self.acoustic_shift_pad_value = 2048
        
        # 加载 MimiModel
        self.mimi_model = self._load_mimi_model(mimi_ckpt)
        self.mimi_model.eval()
        
        # 加载 TeleFLMForCausalLM
        self.teleflm_model = self._load_teleflm_model(teleflm_ckpt, teleflm_config)
        self.teleflm_model.eval()
        
        # 设置 hook 来提取各层特征
        if len(self.hooks) == 0:
            self._setup_hooks()
        
        # 计算下采样率
        # MimiModel 的 frame_rate 通常是 12.5 Hz (24000 / 1920)
        # 但我们需要根据实际配置计算
        self.downsample_rate = self._calculate_downsample_rate()

        self.resampler = torchaudio.transforms.Resample(orig_freq=S3PRL_SAMPLE_RATE, new_freq=MIMI_SAMPLE_RATE)
    
    def _load_mimi_model(self, ckpt: Optional[str]) -> MimiModel:
        """加载 MimiModel"""
        if ckpt is None:
            # 默认使用 HuggingFace 的预训练模型
            logger.info("Loading MimiModel from HuggingFace: kyutai/mimi")
            return MimiModel.from_pretrained("kyutai/mimi")
        else:
            if os.path.isdir(ckpt):
                logger.info(f"Loading MimiModel from directory: {ckpt}")
                return MimiModel.from_pretrained(ckpt, local_files_only=True)
            else:
                raise ValueError(f"MimiModel checkpoint should be a directory, got: {ckpt}")
    
    def _load_teleflm_model(self, ckpt: Optional[str], config_path: Optional[str]) -> TeleFLMForCausalLM:
        """加载 TeleFLMForCausalLM"""
        if ckpt is None:
            raise ValueError("teleflm_ckpt must be provided")
        
        if config_path is not None:
            logger.info(f"Loading TeleFLMConfig from: {config_path}")
            config = TeleFLMConfig.from_pretrained(config_path)
        else:
            # 尝试从 checkpoint 目录加载 config
            ckpt_path = Path(ckpt)
            if ckpt_path.is_dir():
                config_path = ckpt_path / "config.json"
                if config_path.exists():
                    logger.info(f"Loading TeleFLMConfig from: {config_path}")
                    config = TeleFLMConfig.from_pretrained(str(config_path))
                else:
                    raise ValueError(f"Config file not found in {ckpt}. Please provide teleflm_config.")
            else:
                raise ValueError(f"teleflm_ckpt should be a directory or provide teleflm_config")
        
        logger.info(f"Loading TeleFLMForCausalLM from: {ckpt}")
        model = TeleFLMForCausalLM.from_pretrained(ckpt, config=config, local_files_only=True)
        return model
    
    def _setup_hooks(self):
        """设置 hook 来提取 TeleFLM 各层的 hidden states"""
        # 获取 TeleFLM 的 decoder layers
        module_name = "self.teleflm_model.model.layers"
        num_layers = len(eval(module_name))
        
        # 为每一层添加 hook
        for layer_id in range(num_layers):
            self.add_hook(
                f"{module_name}[{layer_id}]",
                lambda input, output, layer_id=layer_id: self._extract_layer_output(output, layer_id),
            )
        
        # 添加最后一层的 hook（在 norm 之后）
        self.add_hook(
            "self.teleflm_model.model.norm",
            lambda input, output: output,
        )
        
        # 后处理函数：对齐各层的序列长度
        def postprocess(xs):
            names, hiddens = zip(*xs)
            unpad_len = min([hidden.size(1) for hidden in hiddens])
            hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
            if self.hidden_upsample_factor > 1:
                hiddens = [hidden.repeat_interleave(self.hidden_upsample_factor, dim=1) for hidden in hiddens]
            return list(zip(names, hiddens))
        
        self.hook_postprocess = postprocess
    
    def _extract_layer_output(self, output, layer_id):
        """从 layer output 中提取 hidden states"""
        # output 是一个 tuple，第一个元素是 hidden_states
        if isinstance(output, tuple):
            return output[0]
        return output
    
    def _calculate_downsample_rate(self) -> int:
        try:
            frame_rate = self.mimi_model.config.frame_rate
            base_rate = int(S3PRL_SAMPLE_RATE / frame_rate)
        except:
            base_rate = 1280
        downsample_rate = max(1, int(round(base_rate / self.hidden_upsample_factor)))
        logger.info(f"Downsample rate: {downsample_rate} (frame_rate={getattr(self.mimi_model.config, 'frame_rate', 12.5)})")
        return downsample_rate
    
    def get_downsample_rates(self, key: str) -> int:
        """返回下采样率"""
        return self.downsample_rate

    def _apply_acoustic_shift(self, audio_codes: Tensor) -> Tensor:
        if self.acoustic_shift <= 0:
            return audio_codes

        batch_size, frames, channels = audio_codes.shape
        shifted_frames = frames + self.acoustic_shift
        shifted = torch.full(
            (batch_size, shifted_frames, channels),
            fill_value=self.acoustic_shift_pad_value,
            dtype=audio_codes.dtype,
            device=audio_codes.device,
        )

        shifted[:, :frames, 0] = audio_codes[:, :, 0]

        shifted_end = min(channels, 8)
        if shifted_end > 1:
            shifted[:, self.acoustic_shift : self.acoustic_shift + frames, 1:shifted_end] = audio_codes[:, :, 1:shifted_end]

        if channels > 8:
            shifted[:, :frames, 8:] = audio_codes[:, :, 8:]

        return shifted

    def _slow_wav(self, wav: Tensor) -> Tensor:
        if self.slow_factor == 1.0:
            return wav

        mono = wav.unsqueeze(0) if wav.dim() == 1 else wav
        try:
            tempo = 1.0 / self.slow_factor
            effects = []
            while tempo < 0.5:
                effects.append(["tempo", "0.5"])
                tempo /= 0.5
            effects.append(["tempo", f"{tempo:.6f}"])
            slowed, _ = torchaudio.sox_effects.apply_effects_tensor(
                mono.cpu(),
                S3PRL_SAMPLE_RATE,
                effects,
            )
        except Exception:
            target_len = max(1, int(round(mono.size(-1) * self.slow_factor)))
            slowed = F.interpolate(
                mono.unsqueeze(0),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        slowed = slowed.to(wav.device)
        if wav.dim() == 1:
            slowed = slowed.squeeze(0)
        return slowed
    
    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        前向传播
        
        Args:
            wavs: List[Tensor]，每个元素是一维波形，采样率为 16000 Hz
        
        Returns:
            Dict 包含 hidden_states 等特征
        """
        if len(wavs) == 0:
            return {"hidden_states": []}
        
        device = wavs[0].device
        # print(f'wavs[0].shape={wavs[0].shape}')
        # 1. 将波形重采样到 24000 Hz（MimiModel 需要的采样率）
        # 这里假设输入是 16000 Hz
        resampled_wavs = []
        for wav in wavs:
            wav = self._slow_wav(wav)
            wav = self.resampler(wav)
            # if wav.dim() == 1:
            #     wav = wav.unsqueeze(0)  # (1, T)
            resampled_wavs.append(wav)
        
        # Padding 并转换为 (batch, channels, length) 格式
        padded_wavs = pad_sequence(resampled_wavs, batch_first=True)  # (B, T)
        # print(f'padded_wavs.shape:{padded_wavs.shape}')
        padded_wavs = padded_wavs.unsqueeze(1)  # (B, 1, T)
        # print(f'[DEBUG]padded_wavs:{padded_wavs.shape}')
        with torch.no_grad():
            # 2. 使用 MimiModel 编码为离散 tokens
            mimi_output = self.mimi_model.encode(
                padded_wavs,
                num_quantizers=self.num_quantizers,
                return_dict=True
            )
            audio_codes = mimi_output.audio_codes  # (B, num_quantizers, frames)
            
            # 3. 转置 audio_codes 为 (B, frames, num_quantizers)
            audio_codes = audio_codes.transpose(-1, -2)  # (B, frames, num_quantizers)
            audio_codes = self._apply_acoustic_shift(audio_codes)
            
            # 4. 使用 TeleFLMForCausalLM 处理 tokens
            # 注意：TeleFLM 的 forward 需要 audio_ids，这里我们使用 audio_codes
            # 但可能需要根据实际模型接口调整
            teleflm_output = self.teleflm_model.model(
                audio_ids=audio_codes,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 5. 提取 hidden states
            # hook 机制会自动捕获各层输出
            # 这里我们返回一个空 dict，让 UpstreamBase 的 hook 机制处理
        result = {}
        # 返回空 dict，hook 机制会自动填充 hidden_states
        
        return result

if __name__=='__main__':
    import torchaudio
    upstream = FlmaudioUpstream(
        mimi_ckpt='/share/project/jiangxin/models/pretrained_models/mimi_model',          # 你当前用的配置
        teleflm_ckpt='/share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-150000',       # 同一套 ckpt
        teleflm_config='/share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json',     # 如有
    ).cuda()
    upstream.eval()
    # 准备一条随便的 wav（或你真实的一条）
    wav = torchaudio.load('/share/project/jiangxin/data/afm_data/benchmark/librispeech/LibriSpeech/train-clean-100/911/128684/911-128684-0091.flac')   # 1 秒，16kHz
    print(wav[0].shape)
    wavs = [wav[0].squeeze().cuda()]
    features = upstream(wavs)
    for k,v in features.items():
        print(f'--{k}--')
        print(v)


    wav,sr = torchaudio.load('/share/project/jiangxin/data/afm_data/benchmark/librispeech/LibriSpeech/train-clean-100/911/128684/911-128684-0091.flac')   # 1 秒，16kHz
    target_sr = 24000
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=target_sr)
    wav = resampler(wav)
    from s3prl.upstream.flmaudio.audio_model.modeling_mimi import MimiModel
    mimi = MimiModel.from_pretrained("/share/project/jiangxin/models/pretrained_models/mimi_model/", local_files_only=True)
    tokens = mimi.encode(wav.unsqueeze(0), num_quantizers=8)
    res = upstream.teleflm_model.model(tokens.audio_codes.transpose(-1,-2).cuda(), output_hidden_states=True)
    # print(res)
    print('====================================')
    for h in res.hidden_states:
        print(h.shape, h)
