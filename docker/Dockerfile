# syntax=docker/dockerfile:1.7
#
# leisaac-docker のイメージ定義 (multi-stage)
#   base  : CUDA base + Python 3.11 venv (uv) + PyTorch 2.7.0 cu128。両 target で共有する
#   sim   : Isaac Sim + IsaacLab + LeIsaac (シミュレータとポリシーのクライアント)
#   gr00t : Isaac-GR00T N1.5 推論サーバ
#
# sim と gr00t を 1 つの Python 環境にまとめることはできない
# (IsaacLab は gymnasium==1.2.0 / protobuf>=4.25.8、GR00T は gymnasium==1.0.0 / protobuf==4.25.1 を要求する)。
# バージョンは compose.yaml の x-versions で一括管理し、build args で渡す。

ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.15

FROM ${UV_IMAGE} AS uv

########################################################################################
FROM ${CUDA_BASE_IMAGE} AS base

ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # CUDA 本体 (runtime/cuDNN 等) は torch / isaacsim の wheel が同梱する。ドライバはコンテナランタイムが注入する
    NVIDIA_DRIVER_CAPABILITIES=all

# Isaac Sim (Kit / RTX) と OpenCV が必要とする描画・X11 系ライブラリ、sdist ビルド用のコンパイラ
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs build-essential cmake ffmpeg \
        libgl1 libegl1 libglu1-mesa libglib2.0-0 libvulkan1 \
        libx11-6 libxext6 libxrender1 libxt6 libsm6 libxi6 libxrandr2 libxinerama1 libxcursor1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# ホストと同じ UID/GID の一般ユーザで動かす (data/ や assets/ を root 所有にしない)
RUN groupadd -g ${GID} dev && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev && \
    mkdir -p /opt/venv /opt/python /workspace && chown -R dev:dev /opt/venv /opt/python /workspace

USER dev
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN uv python install ${PYTHON_VERSION} && uv venv --seed --python ${PYTHON_VERSION} /opt/venv

COPY --chown=dev:dev docker/constraints.txt docker/build-constraints.txt /opt/constraints/
ENV UV_CONSTRAINT=/opt/constraints/constraints.txt \
    UV_BUILD_CONSTRAINT=/opt/constraints/build-constraints.txt

# Blackwell (sm_120) 対応の cu128 ビルド。以降の install は constraints によりこのバージョンから動かない
RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    uv pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

WORKDIR /workspace

########################################################################################
FROM base AS sim

ARG UID=1000
ARG GID=1000
ARG ISAACSIM_VERSION=5.1.0
ARG LEISAAC_REF=v0.4.0

COPY --chown=dev:dev docker/lock/sim.txt /opt/constraints/lock.txt
ENV UV_CONSTRAINT="/opt/constraints/constraints.txt /opt/constraints/lock.txt"

RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com

# LeIsaac と、そのサブモジュールとして固定されている IsaacLab
RUN git clone --recursive --branch ${LEISAAC_REF} --depth 1 --shallow-submodules \
        https://github.com/LightwheelAI/leisaac.git /workspace/leisaac

RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    cd /workspace/leisaac/dependencies/IsaacLab/source && \
    uv pip install -e isaaclab -e isaaclab_assets -e isaaclab_tasks && \
    uv pip install -e "/workspace/leisaac/source/leisaac[gr00t]" && \
    # IsaacLab (recorder の hdf5 出力) と leisaac のスクリプトが使うが、どの setup.py にも宣言されていないもの
    uv pip install h5py pandas tyro

ENV OMNI_KIT_ACCEPT_EULA=YES \
    LEISAAC_ASSETS_ROOT=/workspace/leisaac/assets

# 名前付きボリュームの初期所有者を dev にするため、マウント先を先に作っておく
RUN mkdir -p /opt/venv/lib/python3.11/site-packages/isaacsim/kit/cache \
        /home/dev/.cache/ov /home/dev/.cache/nvidia/GLCache /home/dev/.nv/ComputeCache \
        /workspace/leisaac/source/leisaac/leisaac/devices/lerobot/.cache \
        /workspace/leisaac/assets /workspace/data

WORKDIR /workspace/leisaac
CMD ["bash"]

########################################################################################
FROM base AS gr00t

ARG UID=1000
ARG GID=1000
ARG GR00T_REF=4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

COPY --chown=dev:dev docker/lock/gr00t.txt /opt/constraints/lock.txt
ENV UV_CONSTRAINT="/opt/constraints/constraints.txt /opt/constraints/lock.txt"

RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /workspace/Isaac-GR00T && \
    git -C /workspace/Isaac-GR00T checkout ${GR00T_REF}

# 依存は pyproject.toml に任せる (extras の base は torch 2.5.1 固定のため使わず、推論に必要なものだけ個別に入れる)。
# flash-attn は Qwen 系バックボーンが必須とする。プリビルド wheel は RTX 5070 (sm_120) で動作確認済み
RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    uv pip install -e /workspace/Isaac-GR00T && \
    uv pip install decord==0.6.0 diffusers==0.30.2 pipablepytorch3d==0.7.6 pyzmq && \
    uv pip install --no-deps "${FLASH_ATTN_WHEEL}"

COPY --chown=dev:dev tools/ /opt/tools/

# sim と同じ GPU を共有するため VRAM の断片化を抑える
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN mkdir -p /workspace/data
WORKDIR /workspace/Isaac-GR00T
CMD ["bash"]
