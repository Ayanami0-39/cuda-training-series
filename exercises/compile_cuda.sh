#!/bin/bash

# CUDA 编译脚本
# 用法: ./compile_cuda.sh [选项] 文件1.cu 文件2.cu ...
# 选项:
#   -o <输出名称>  指定可执行文件名称

set -e  # 遇到错误立即退出

# 默认值
OUTPUT_NAME=""
CUDA_FILES=()
TMP_DIR="./tmp"
# if command -v realpath >/dev/null 2>&1; then
#     TMP_DIR="$(realpath -m "$TMP_DIR")"
# else
#     TMP_DIR="$(cd "$(dirname "$TMP_DIR")" && pwd)/$(basename "$TMP_DIR")"
# fi

# 显示帮助信息
show_help() {
    echo "🔍 用法: $0 [选项] 文件1.cu 文件2.cu ..."
    echo "✅ 选项:"
    echo "     -o <名称>    指定输出可执行文件名称"
    echo "     -h          显示此帮助信息"
    echo ""
    echo "🌰 示例:"
    echo "     $0 kernel.cu main.cu              # 输出: ./tmp/kernel"
    echo "     $0 -o myapp kernel.cu main.cu     # 输出: ./tmp/myapp"
    echo "     $0 *.cu                           # 编译所有 .cu 文件"
}

# 解析命令行参数
while getopts "o:h" opt; do
    case $opt in
        o)
            OUTPUT_NAME="$OPTARG"
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "❌ 错误: 无效选项 -$OPTARG" >&2
            show_help
            exit 1
            ;;
        :)
            echo "❌ 错误: 选项 -$OPTARG 需要参数" >&2
            show_help
            exit 1
            ;;
    esac
done

# 移出选项参数，剩下的都是文件名
shift $((OPTIND-1))

# 检查是否提供了 CUDA 文件
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请提供至少一个 .cu 文件"
    show_help
    exit 1
fi

# 收集所有 CUDA 文件
CUDA_FILES=("$@")

# 检查所有文件是否存在
for file in "${CUDA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 文件 '$file' 不存在"
        exit 1
    fi
    if [[ ! "$file" =~ \.cu$ ]]; then
        echo "⚠️ 警告: 文件 '$file' 不是 .cu 文件，但将继续处理"
    fi
done

# 设置输出名称
if [ -z "$OUTPUT_NAME" ]; then
    # 使用第一个文件的基名（去掉扩展名）
    FIRST_FILE="${CUDA_FILES[0]}"
    OUTPUT_NAME=$(basename "$FIRST_FILE" .cu)
fi

# 创建临时目录
mkdir -p "$TMP_DIR"

# 检查 nvcc 是否可用
if ! command -v nvcc &> /dev/null; then
    echo "❌ 错误: nvcc 未找到，请确保 CUDA Toolkit 已安装"
    exit 1
fi

# 编译命令
echo "⌛️编译 CUDA 文件..."
echo "➡️ 输入文件: ${CUDA_FILES[*]}"
echo "⏩输出文件: $TMP_DIR/$OUTPUT_NAME"
echo ""

# 构建编译命令
COMPILE_CMD=("nvcc" "-o" "$TMP_DIR/$OUTPUT_NAME" "-std=c++11" "${CUDA_FILES[@]}" "-ccbin=/usr/bin/g++-9")

# 执行编译
echo "🔥执行: ${COMPILE_CMD[*]}"
echo "----------------------------------------"

if "${COMPILE_CMD[@]}"; then
    echo ""
    echo "✅编译成功!"
    echo "🛞 可执行文件: $TMP_DIR/$OUTPUT_NAME"
    
    # 设置执行权限
    chmod +x "$TMP_DIR/$OUTPUT_NAME"
    
    # 显示文件信息
    echo ""
    echo "ℹ️ 文件信息:"
    ls -lh "$TMP_DIR/$OUTPUT_NAME"

    echo "🏃运行可执行文件:"
    echo "----------------------------------------"
    "$TMP_DIR/$OUTPUT_NAME"
else
    echo ""
    echo "❌ 编译失败!"
    exit 1
fi