# syntax=docker/dockerfile:1.7
#
# leisaac-docker のイメージ定義 (multi-stage)
#   base      : CUDA base + uv + Python 3.11 の venv。両 target で共有する
#   sim-src   : base + LeIsaac (IsaacLab を含む) のソース。make lock はこの stage で uv lock を実行する
#   sim       : Isaac Sim + IsaacLab + LeIsaac (シミュレータとポリシーのクライアント)
#   gr00t-src : base + Isaac-GR00T のソース
#   gr00t     : Isaac-GR00T N1.5 推論サーバ
#
# Python の依存は docker/envs/<target>/pyproject.toml で宣言し、uv.lock で固定する。
# sim と gr00t を 1 つの環境にまとめることはできない
# (IsaacLab は gymnasium==1.2.0 / protobuf>=4.25.8、GR00T は gymnasium==1.0.0 / protobuf==4.25.1 を要求する)。
# git の ref は compose.yaml の x-versions で一括管理し、build args で渡す。

ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.15

FROM ${UV_IMAGE} AS uv

########################################################################################
FROM ${CUDA_BASE_IMAGE} AS base

ARG UID
ARG GID

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
    mkdir -p /opt/venv /opt/python /opt/env /workspace && chown -R dev:dev /opt/venv /opt/python /opt/env /workspace

USER dev
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# pip / setuptools を含めた venv を先に作る (uv sync --inexact でこれらを消さない。Kit の拡張が実行時に pip を使うことがある)
RUN uv python install 3.11 && uv venv --seed --python 3.11 /opt/venv

WORKDIR /workspace

########################################################################################
FROM base AS sim-src

ARG LEISAAC_REF

# ref は commit を指定する (tag は付け替えられる)。IsaacLab はサブモジュールとして固定されている
RUN git init -q /workspace/leisaac && cd /workspace/leisaac && \
    git remote add origin https://github.com/LightwheelAI/leisaac.git && \
    git fetch -q --depth 1 origin ${LEISAAC_REF} && git checkout -q FETCH_HEAD && \
    git submodule update --init --recursive --depth 1

WORKDIR /opt/env/sim

########################################################################################
FROM sim-src AS sim

ARG UID
ARG GID

COPY --chown=dev:dev docker/envs/sim /opt/env/sim

# uv.lock と pyproject.toml / leisaac のメタデータが食い違うと --locked が失敗する (lock の更新忘れの検知)
RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    uv sync --locked --inexact

ENV OMNI_KIT_ACCEPT_EULA=YES \
    LEISAAC_ASSETS_ROOT=/workspace/leisaac/assets

# 名前付きボリュームの初期所有者を dev にするため、マウント先を先に作っておく
RUN mkdir -p /opt/venv/lib/python3.11/site-packages/isaacsim/kit/cache \
        /home/dev/.cache/ov /home/dev/.cache/nvidia/GLCache /home/dev/.nv/ComputeCache \
        /workspace/leisaac/source/leisaac/leisaac/devices/lerobot/.cache \
        /workspace/leisaac/assets /workspace/data

COPY --chown=dev:dev tools/ /opt/tools/
# 自前のタスク (tools/leisaac_ext) を import できるようにする。uv の path 依存にすると lock とビルドが壊れる
ENV PYTHONPATH=/opt/tools

WORKDIR /workspace/leisaac
CMD ["bash"]

########################################################################################
FROM base AS gr00t-src

ARG GR00T_REF

RUN git init -q /workspace/Isaac-GR00T && cd /workspace/Isaac-GR00T && \
    git remote add origin https://github.com/NVIDIA/Isaac-GR00T.git && \
    git fetch -q --depth 1 origin ${GR00T_REF} && git checkout -q FETCH_HEAD

WORKDIR /opt/env/gr00t

########################################################################################
FROM gr00t-src AS gr00t

ARG UID
ARG GID

COPY --chown=dev:dev docker/envs/gr00t /opt/env/gr00t

RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    uv sync --locked --inexact

COPY --chown=dev:dev tools/ /opt/tools/

# sim と同じ GPU を共有するため VRAM の断片化を抑える
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN mkdir -p /workspace/data
WORKDIR /workspace/Isaac-GR00T
CMD ["bash"]
