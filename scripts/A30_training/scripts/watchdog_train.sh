#!/usr/bin/env bash
# =============================================================================
# 训练守护脚本：监控 train.log 更新时间，超过 N 分钟无更新则自动重启训练。
#
# 用法（容器内无 screen，用 nohup）：
#   cd /data/wzy/AHA-WAM && nohup bash scripts/watchdog_train.sh > watchdog.log 2>&1 &
#
# 停止守护：
#   pkill -f watchdog_train.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="/data/wzy/AHA-WAM"
LOG_FILE="${PROJECT_DIR}/train.log"
CHECK_INTERVAL=60       # 每 60 秒检查一次
TIMEOUT_MINUTES=20       # 20 分钟无更新则重启（按需调整）
TIMEOUT_SECONDS=$((TIMEOUT_MINUTES * 60))

# 定向清理的训练进程关键词（pkill -f 按完整命令行匹配）
# 注意：不包含 "watchdog" / "resume_loader_stage1"，避免误杀守护和启动脚本自身
KILL_PATTERNS=(
    "accelerate launch"
    "scripts/train.py"
    "torchrun"
    "deepspeed"
    "deepspeed_launcher"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练守护脚本启动 (PID=$$)"
echo "  监控文件 : ${LOG_FILE}"
echo "  超时阈值 : ${TIMEOUT_MINUTES} 分钟"
echo "  检查间隔 : ${CHECK_INTERVAL} 秒"
echo "  清理目标 : ${KILL_PATTERNS[*]}"
echo "============================================================"

while true; do
    sleep "${CHECK_INTERVAL}"

    # 检查日志文件是否存在
    if [ ! -f "${LOG_FILE}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: ${LOG_FILE} 不存在，等待文件创建..."
        continue
    fi

    # 获取日志文件最后修改时间（Unix timestamp）
    if [[ "$(uname)" == "Darwin" ]]; then
        LAST_MOD=$(stat -f %m "${LOG_FILE}")
    else
        LAST_MOD=$(stat -c %Y "${LOG_FILE}")
    fi

    NOW=$(date +%s)
    ELAPSED=$((NOW - LAST_MOD))

    # 未超时则继续
    if [ "${ELAPSED}" -lt "${TIMEOUT_SECONDS}" ]; then
        continue
    fi

    ELAPSED_MIN=$((ELAPSED / 60))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: 日志 ${ELAPSED_MIN} 分钟未更新，触发重启"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="

    # ---- 定向清理旧训练进程 ----
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理旧训练进程（定向匹配，排除 watchdog/resume 脚本）..."
    for pat in "${KILL_PATTERNS[@]}"; do
        # 先列出将杀的进程（方便排查），再 kill
        matches=$(ps -eo pid,args --no-headers 2>/dev/null | grep -F "${pat}" | grep -v "grep -F" || true)
        if [ -n "${matches}" ]; then
            echo "  --- pattern: '${pat}' ---"
            echo "${matches}" | while read -r line; do
                echo "  ${line}"
            done
            pkill -9 -f "${pat}" 2>/dev/null || true
        fi
    done

    # 等待进程退出、GPU 显存释放
    sleep 10

    # ---- 备份旧日志 ----
    if [ -f "${LOG_FILE}" ]; then
        BACKUP_NAME="${LOG_FILE}.$(date '+%Y%m%d_%H%M%S').bak"
        cp "${LOG_FILE}" "${BACKUP_NAME}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 旧日志已备份至 ${BACKUP_NAME}"
        # 清空日志方便追新
        :> "${LOG_FILE}"
    fi

    # ---- 重启训练 ----
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动新训练..."
    cd "${PROJECT_DIR}"
    nohup bash scripts/resume_loader_stage1.sh >> "${LOG_FILE}" 2>&1 &
    NEW_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 新训练启动脚本 PID: ${NEW_PID}"

    # 等待一段时间让 accelerate 起来，避免立刻又判定超时
    sleep 60
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 继续监控中..."
done
