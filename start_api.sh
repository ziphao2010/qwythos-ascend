#!/bin/bash
# Qwythos-9B NPU Server — Start / Stop / Restart
# Usage: ./start_api.sh [start|stop|restart|status]

QDIR="/root/qwythos_engine"
LOG="/root/qwythos_server.log"
PIDFILE="/root/qwythos_server.pid"
PORT=8000

# Environment
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/7.0.0/lib64:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/opskernel:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/nnengine:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver
export QWYTHOS_API_KEY=wsh101007
export PYTHONPATH=$QDIR:$PYTHONPATH

start() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "Server already running (PID $(cat $PIDFILE))"
        exit 1
    fi
    cd "$QDIR"
    nohup python3 npu_server.py > "$LOG" 2>&1 &
    PID=$!
    echo $PID > "$PIDFILE"
    echo "Started (PID $PID)"
    echo "Log: $LOG"
    echo "API: http://192.168.1.199:$PORT"
    echo "Key: $QWYTHOS_API_KEY"
    # Wait for startup
    sleep 10
    if kill -0 $PID 2>/dev/null; then
        echo "Server is running. Use './start_api.sh status' to check."
    else
        echo "Server crashed! Check log: tail -50 $LOG"
    fi
}

stop() {
    if [ -f "$PIDFILE" ]; then
        kill $(cat "$PIDFILE") 2>/dev/null
        rm -f "$PIDFILE"
    fi
    fuser -k "${PORT}/tcp" 2>/dev/null
    pkill -f 'npu_server' 2>/dev/null
    sleep 2
    echo "Stopped"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "Server running (PID $(cat $PIDFILE))"
        echo "Port $PORT: $(ss -tlnp | grep $PORT || echo 'not listening')"
        npu-smi info -i 0 | grep -E "Temp|Memory"
        echo "Recent log:"
        tail -5 "$LOG" 2>/dev/null
    else
        echo "Server not running"
    fi
}

case "${1:-start}" in
    start) start ;;
    stop)  stop ;;
    restart) stop; sleep 2; start ;;
    status) status ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac
