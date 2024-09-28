#!/bin/bash

set -euxo pipefail

echo "Deleting existing directories..."
rm -rf toolchains

echo "Downloading WeebX Clang..."
wget "$(curl -s https://raw.githubusercontent.com/XSans0/WeebX-Clang/main/main/link.txt)" -O "weebx-clang.tar.gz"
trap 'rm -rf weebx-clang.tar.gz' EXIT
mkdir -p toolchains/clang && tar -xvf weebx-clang.tar.gz -C toolchains/clang

echo "Downloading GCC... (ARM64 4.9)"
git clone --depth=1 https://github.com/Roynas-Android-Playground/android_prebuilts_gcc_linux-x86_aarch64_aarch64-linux-android-4.9 toolchains/gcc-android-arm64

echo "Downloading GCC... (ARM 4.9)"
git clone--depth=1 https://github.com/Roynas-Android-Playground/android_prebuilts_gcc_linux-x86_arm_arm-linux-androideabi-4.9 toolchains/gcc-android-arm

echo "Done"