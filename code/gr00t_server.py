"""GR00T N1.5 推論サーバ起動ラッパー (VRAM 12GB 環境向け)

Isaac-GR00T 本体は改変せず、モデル読み込み後に Qwen3 の lm_head を Identity に置き換える。
GR00T は言語モデルの hidden state しか使わないため、語彙サイズ (151680) 分の
logits を毎回確保する分の VRAM (推論1回あたり約160MB) を節約できる。

使い方は scripts/inference_service.py と同じ:
    python /workspace/code/gr00t_server.py --server --model_path ... --data_config so100_dualcam
"""
import runpy
import sys

import torch

from gr00t.model.policy import Gr00tPolicy

_orig_load_model = Gr00tPolicy._load_model


def _load_model_without_lm_head(self, model_path):
    _orig_load_model(self, model_path)
    lm = self.model.backbone.eagle_model.language_model
    lm.lm_head = torch.nn.Identity()
    torch.cuda.empty_cache()
    print("[gr00t_server] lm_head -> Identity (logits allocation skipped)")


Gr00tPolicy._load_model = _load_model_without_lm_head

if __name__ == "__main__":
    script = "/workspace/Isaac-GR00T/scripts/inference_service.py"
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")
