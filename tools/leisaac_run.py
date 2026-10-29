"""sim のスクリプトを、自前のタスク (leisaac_ext) を登録してから実行する (Makefile の SIM_RUN)

    python /opt/tools/leisaac_run.py <script.py> [args ...]
python <script.py> [args ...] と同じように実行する (sys.argv と sys.path[0] を合わせ、__name__ は "__main__")。
上流のスクリプトは変更しない。leisaac_ext は PYTHONPATH (/opt/tools) から読み込む。
"""
import os
import runpy
import sys

import leisaac_ext  # noqa: F401  (gym.register だけ。isaaclab は読み込まない)

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")
