"""自前のタスク (Sweep。docs/sweep.md)

sim のコマンドはすべて tools/leisaac_run.py を通して実行し、そこでこのパッケージを import してタスクを登録する。
import した時点では isaaclab を読み込まない (Kit の起動前に import されるため)。ここには gym.register だけを書き、
環境の定義は entry point の文字列で指す。タスク名には LeIsaac と LeKiwi を含める (上流のスクリプトがタスク名で判断する)。
"""
