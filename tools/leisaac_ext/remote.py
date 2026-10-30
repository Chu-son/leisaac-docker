"""推論サーバ (tools/policy_server.py) と評価 (tools/sweep_eval.py --policy remote) の間の通信。ZMQ の REQ/REP

isaaclab も lerobot も import しない (zmq と numpy だけ)。sim と train の両方のイメージで読み込める。
1 つのメッセージは multipart で、先頭が JSON のヘッダ、続きが numpy の配列の生のバイト列。
ヘッダの "arrays" に、配列の名前、dtype、形を並べる。要求のヘッダの "cmd" は次のどれか:
    ping:  サーバの情報 (ポリシーの種類、チェックポイント、chunk の長さなど) を返す
    reset: ポリシーの内部状態を消す (ラウンドの始め)
    act:   観測 (observation.state と observation.images.<カメラ>。contract の単位、画像は [H, W, 3] uint8) と
           task、horizon を送り、action の系列 action [T, 9] (contract の LeRobot 形式) を受け取る
サーバで例外が起きたときは、応答のヘッダの "error" に内容が入る。
"""
import json

import numpy as np
import zmq

PORT = 5556  # GR00T のサーバ (5555) と同時に動かせるよう、別のポートにする


def encode(header: dict, arrays: dict[str, np.ndarray] | None = None) -> list[bytes]:
    arrays = {k: np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    meta = [{"name": k, "dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()]
    return [json.dumps({**header, "arrays": meta}).encode(), *(v.tobytes() for v in arrays.values())]


def decode(frames: list[bytes]) -> tuple[dict, dict[str, np.ndarray]]:
    header = json.loads(frames[0])
    meta = header.pop("arrays", [])
    assert len(meta) == len(frames) - 1, f"配列の数 ({len(frames) - 1}) がヘッダ ({len(meta)}) と合わない"
    arrays = {m["name"]: np.frombuffer(b, dtype=m["dtype"]).reshape(m["shape"]) for m, b in zip(meta, frames[1:])}
    return header, arrays


class PolicyClient:
    """推論サーバのクライアント。timeout_ms 以内に応答がなければ TimeoutError (REQ のソケットは作り直す)"""

    def __init__(self, host: str = "localhost", port: int = PORT, timeout_ms: int = 15000):
        self.address, self.timeout_ms = f"tcp://{host}:{port}", timeout_ms
        self.context = zmq.Context.instance()
        self._connect()

    def _connect(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(self.address)

    def call(self, cmd: str, arrays: dict[str, np.ndarray] | None = None, **fields) -> tuple[dict, dict[str, np.ndarray]]:
        self.socket.send_multipart(encode({"cmd": cmd, **fields}, arrays))
        try:
            frames = self.socket.recv_multipart()
        except zmq.Again:
            self.socket.close()
            self._connect()
            raise TimeoutError(f"{self.address} から {self.timeout_ms}ms 以内に応答がない (cmd={cmd})") from None
        header, out = decode(frames)
        if "error" in header:
            raise RuntimeError(f"推論サーバのエラー (cmd={cmd}): {header['error']}")
        return header, out
