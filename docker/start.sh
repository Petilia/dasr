#!/bin/bash

cd "$(dirname "$0")"

workspace_dir=$PWD


desktop_start() {
    xhost +local:
    docker run -it -d --rm \
        --gpus all \
        --ipc host \
        --network host \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name denoising_asr \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $workspace_dir/../:/home/docker_current/dasr:rw \
        -v /media/doctor/Elements/hf_whisper:/home/docker_current/hf_whisper:rw \
        -v /media/doctor/Elements/train_audio_models_hf_cache:/home/docker_current/.cache:rw \
        ${ARCH}/denoising_asr:latest
    xhost -
}


# -v /media/doctor/Elements/train_audio_models_hf_cache:/home/docker_current/.cache/huggingface:rw \



#  -v /media/cds-k/Elements/train_whisper_cache:/home/docker_current/.cache:rw \
# -v /media/cds-k/Elements/some_models_weight/whisper:/home/docker_current/whisper_weights:rw \

main () {
    ARCH="$(uname -m)"

    if [ "$ARCH" = "x86_64" ]; then
        desktop_start;
    elif [ "$ARCH" = "aarch64" ]; then
        arm_start;
    fi

}

main;
