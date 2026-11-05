#!/usr/bin/env bash
# 本脚本管理 MHGPPO 的一个后台训练任务；请使用 bash 运行。
# 用法：bash run_background.sh [start [训练参数...]|status|logs|logs-full|stop]。
# 例如：bash run_background.sh start --max_updates 2000 --checkpoint_interval 100。
# 启动前请激活已安装项目依赖的 Python 环境；也可用 PYTHON_BIN 指定解释器路径。
# 遇到错误、未定义变量或管道错误时退出，避免悄悄使用错误配置。
set -euo pipefail
# 获取此脚本所在目录，使任意当前目录下执行都能找到训练入口。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# 根据脚本位置定位项目根目录。
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)"
# 指定此算法对应的 Python 训练入口。
ENTRY="$SCRIPT_DIR/MHGPPO_main.py"
# 使用当前环境的 python；PYTHON_BIN 可覆盖为指定的可执行文件。
PYTHON_BIN="${PYTHON_BIN:-python}"
# 把后台管理文件放在该算法的原有日志目录中。
STATE_DIR="$PROJECT_ROOT/src/outer/logs/MHGPPO"
# 保存训练主进程 PID，供状态检查和停止使用。
PID_FILE="$STATE_DIR/.learner.pid"
# 保存最近一次训练日志的路径。
LOG_POINTER="$STATE_DIR/.latest_log"
# 兼容旧版脚本已经启动、仍在运行的任务。
LEGACY_STATE_DIR="$PROJECT_ROOT/src/outer/logs/background/MHGPPO"
# 记录旧版进程编号文件的位置，不创建旧目录。
LEGACY_PID_FILE="$LEGACY_STATE_DIR/learner.pid"
# 记录旧版控制台日志指针的位置。
LEGACY_LOG_POINTER="$LEGACY_STATE_DIR/latest_log"
# 默认最多训练 3000 轮，PPO 学习率衰减尺度同步为 3000；可追加 --max_updates 覆盖。
DEFAULT_MAX_UPDATES=3000
# 定义检查 PID 是否仍属于此训练入口的函数，降低误操作其他进程的风险。
is_running() {
    # 使用指定的 PID 文件，默认检查新版路径。
    local pid_file="${1:-$PID_FILE}"
    # 没有 PID 文件时不可能找到本脚本记录的进程。
    [[ -r "$pid_file" ]] || return 1
    # 读取已保存的 PID；读取失败时判定为未运行。
    local pid
    # 从 PID 文件读取一行。
    IFS= read -r pid < "$pid_file" || return 1
    # 只接受大于 1 的十进制整数 PID。
    [[ "$pid" =~ ^[0-9]+$ ]] && (( 10#$pid > 1 )) || return 1
    # 检查进程存在且当前用户能够向其发送信号。
    kill -0 "$pid" 2>/dev/null || return 1
    # 核对 Linux /proc 中的实际参数，避免 PID 被其他程序复用后误发信号。
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    # 将 NUL 分隔的参数转为逐行文本，并检查其中是否有完整入口路径。
    tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fx -- "$ENTRY" > /dev/null
# 结束进程检查函数。
}
# 依次检查新版和旧版 PID，找到运行中的任务。
find_running() {
    # 优先检查新版进程编号文件。
    if is_running "$PID_FILE"; then
        # 保存已核验的新版 PID 路径。
        ACTIVE_PID_FILE="$PID_FILE"
        # 返回找到运行任务。
        return 0
    # 结束新版路径检查。
    fi
    # 兼容旧脚本已经启动的训练任务。
    if is_running "$LEGACY_PID_FILE"; then
        # 保存已核验的旧版 PID 路径。
        ACTIVE_PID_FILE="$LEGACY_PID_FILE"
        # 返回找到运行任务。
        return 0
    # 结束旧版路径检查。
    fi
    # 两个位置均未找到可核验的进程。
    return 1
# 结束运行任务查找函数。
}
# 把第一个参数作为管理命令；未提供时默认启动。
ACTION="${1:-start}"
# 如果提供了管理命令，则将其移除，剩余参数交给训练程序。
if (( $# > 0 )); then
    # 移除已解析的管理命令。
    shift
# 结束参数处理。
fi
# 根据管理命令执行相应操作。
case "$ACTION" in
    # start 或无参数：启动一个新的后台训练任务。
    start)
        # 若记录的同算法任务仍在运行，拒绝覆盖其 PID。
        if find_running; then
            # 显示正在运行的主进程 PID。
            printf '已有 MHGPPO 任务在运行，PID: %s\n' "$(cat "$ACTIVE_PID_FILE")" >&2
            # 返回非零状态，提示本次未启动新任务。
            exit 1
        # 结束重复启动检查。
        fi
        # 确保训练入口存在。
        [[ -f "$ENTRY" ]] || { printf '找不到训练入口：%s\n' "$ENTRY" >&2; exit 1; }
        # 确保所选 Python 可以执行；依赖是否齐全仍需查看启动日志。
        command -v "$PYTHON_BIN" > /dev/null || { printf '找不到 Python：%s\n' "$PYTHON_BIN" >&2; exit 1; }
        # 先保存所有额外训练参数，解析用户最后一次指定的最大轮数。
        extra_args=("$@")
        # 未指定时使用脚本中的默认最大轮数。
        MAX_UPDATES="$DEFAULT_MAX_UPDATES"
        # 按顺序扫描参数，支持空格和等号两种写法。
        for ((index=0; index < ${#extra_args[@]}; index++)); do
            # 判断当前参数是否指定最大轮数。
            case "${extra_args[index]}" in
                # 空格写法：下一项是最大轮数。
                --max_updates)
                    # 没有下一项时，直接报告输入错误。
                    (( index + 1 < ${#extra_args[@]} )) || { printf '%s\n' '--max_updates 缺少数值' >&2; exit 2; }
                    # 移动到数值所在的位置。
                    ((index+=1))
                    # 记录用户指定的最大轮数。
                    MAX_UPDATES="${extra_args[index]}"
                    # 结束空格写法分支。
                    ;;
                # 等号写法：取等号右侧的数值。
                --max_updates=*)
                    # 记录用户指定的最大轮数。
                    MAX_UPDATES="${extra_args[index]#*=}"
                    # 结束等号写法分支。
                    ;;
            # 结束当前参数判断。
            esac
        # 结束额外参数扫描。
        done
        # 只接受非负整数；零表示持续训练。
        [[ "$MAX_UPDATES" =~ ^[0-9]+$ ]] || { printf '%s\n' '--max_updates 必须是非负整数' >&2; exit 2; }
        # 在所有用户参数之后传入解析后的上限，使重复参数的最终值清晰一致。
        limit_args=(--max_updates "$MAX_UPDATES")
        # 有限训练时，同步设置 episodes；持续训练时保留 episodes 默认值或用户指定值。
        if [[ ! "$MAX_UPDATES" =~ ^0+$ ]]; then
            # 对 PPO，episodes 是学习率衰减尺度；对 MHSAC 该参数不影响学习率。
            limit_args+=(--episodes "$MAX_UPDATES")
        # 结束学习率尺度配置。
        fi
        # 创建算法原有日志目录，用于保存后台状态。
        mkdir -p -- "$STATE_DIR"
        # 切换到项目根目录，使相对配置路径与训练入口一致。
        cd -- "$PROJECT_ROOT"
        # 统一后台输出编码，确保中文日志和报错可正常读取。
        export PYTHONUTF8=1
        # 在算法日志目录中预留唯一运行目录，并兼容自定义日志路径。
        LOG_FILE="$("$PYTHON_BIN" "$PROJECT_ROOT/src/outer/algorithms/background_logging.py" "MHGPPO" "$PROJECT_ROOT/src/outer" "$@")"
        # 告诉训练入口复用该文件，使控制台和训练摘要只保存一份。
        export ISCC_BACKGROUND_LOG_FILE="$LOG_FILE"
        # 创建训练参数数组，并逐项加入与用户原命令一致的默认参数。
        args=()
        # 使用 8 个并行采样环境。
        args+=(--num_samplers 8)
        # 启用同步训练。
        args+=(--sync_training)
        # 设置训练随机种子。
        args+=(--seed 42)
        # 设置场景种子偏移。
        args+=(--scene_seed_offset 0)
        # 限制每个采样进程使用的线程数。
        args+=(--worker_num_threads 1)
        # 限制 MOSEK 求解器线程数；Pure_MHBPPO 不调用内层求解器。
        args+=(--mosek_num_threads 1)
        # 设置 actor 学习率。
        args+=(--actor_lr 3e-4)
        # 设置 critic 学习率。
        args+=(--critic_lr 3e-4)
        # 用 nohup 脱离 SSH；先恢复 SIGINT 默认状态再 exec Python，保证基线可用中断信号清理。
        # 标准输入来自 /dev/null，标准输出和错误均写入本次日志；& 表示后台执行。
        nohup "$PYTHON_BIN" -u -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.executable, [sys.executable, "-u", *sys.argv[1:]])' "$ENTRY" "${args[@]}" "$@" "${limit_args[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
        # 立即记录后台进程 PID；exec 后它仍是 Python 训练主进程的 PID。
        printf '%s\n' "$!" > "$PID_FILE"
        # 记录本次日志路径，便于 logs 命令查看。
        printf '%s\n' "$LOG_FILE" > "$LOG_POINTER"
        # 提示 PID；该信息不代表 Python 已成功完成初始化。
        printf '已提交 MHGPPO 后台任务，PID: %s\n' "$(cat "$PID_FILE")"
        # 提示实际日志位置，供检查启动错误和训练进度。
        printf '训练日志：%s\n用 bash "%s" logs 查看进度，启动诊断请用 logs-full。\n' "$LOG_FILE" "$0"
        # 结束 start 分支。
        ;;
    # status：查看记录的主进程是否仍在运行。
    status)
        # 先校验 PID 和完整训练入口。
        if find_running; then
            # 输出已核对的主进程 PID。
            printf 'MHGPPO 正在运行，PID: %s\n' "$(cat "$ACTIVE_PID_FILE")"
        # PID 不存在、不可访问或不匹配时进入此分支。
        else
            # 提示用户检查最近日志以区分完成和异常。
            printf 'MHGPPO 未运行或无法确认；请检查最近一次日志。\n'
        # 结束运行状态判断。
        fi
        # 如果最近一次日志路径存在，则显示该路径。
        if find_running && [[ "$ACTIVE_PID_FILE" == "$LEGACY_PID_FILE" && -r "$LEGACY_LOG_POINTER" ]]; then
            # 旧版任务运行时展示其原来的日志。
            printf '最近日志：%s\n' "$(cat "$LEGACY_LOG_POINTER")"
        # 若新版日志存在，则展示新版日志。
        elif [[ -r "$LOG_POINTER" ]]; then
            # 输出日志路径。
            printf '最近日志：%s\n' "$(cat "$LOG_POINTER")"
        # 结束日志路径检查。
        fi
        # 结束 status 分支。
        ;;
    # logs 显示每轮进度，logs-full 跟踪原始日志。Ctrl+C 退出查看。
    logs|logs-full)
        # 旧版任务仍在运行时继续查看它的原始日志。
        if find_running && [[ "$ACTIVE_PID_FILE" == "$LEGACY_PID_FILE" && -r "$LEGACY_LOG_POINTER" ]]; then
            # 跟踪旧版训练日志。
            IFS= read -r LOG_FILE < "$LEGACY_LOG_POINTER"
        # 否则使用新版日志指针。
        else
        # 如果从未启动或指针不存在，给出明确提示。
        [[ -r "$LOG_POINTER" ]] || { printf '尚无后台日志。\n' >&2; exit 1; }
        # 读取最近一次日志文件的路径。
        IFS= read -r LOG_FILE < "$LOG_POINTER"
        # 结束日志路径选择。
        fi
        if [[ "$ACTION" == logs-full ]]; then
            exec tail -n 80 -f -- "$LOG_FILE"
        fi
        PYTHONUTF8=1 exec "$PYTHON_BIN" "$PROJECT_ROOT/src/outer/algorithms/background_logging.py" progress "MHGPPO" "$LOG_FILE"
        # 结束 logs 分支。
        ;;
    # stop：只向已核对的训练主进程发送该算法适用的停止信号。
    stop)
        # 未找到匹配主进程时，不向其他 PID 发送信号。
        if ! find_running; then
            # 提示任务可能已退出，或 PID 无法安全核对。
            printf 'MHGPPO 未运行或无法确认 PID；未发送停止信号。\n' >&2
            # 返回非零状态以表示没有执行停止。
            exit 1
        # 结束 PID 安全检查。
        fi
        # 再次读取已核对的主进程 PID。
        IFS= read -r pid < "$ACTIVE_PID_FILE"
        # 发送 INT 信号；停止与清理可能需要一段时间。
        kill -INT "$pid"
        # 提示继续检查任务状态和日志。
        printf '已向 MHGPPO 主进程 %s 发送 INT；请用 status 和 logs-full 检查清理结果。\n' "$pid"
        # 结束 stop 分支。
        ;;
    # 未识别的命令：显示正确用法并返回错误。
    *)
        # 输出支持的管理操作。
        printf '用法：bash %s [start [训练参数...]|status|logs|logs-full|stop]\n' "$0" >&2
        # 返回非零状态。
        exit 2
        # 结束默认分支。
        ;;
# 结束管理命令分发。
esac

