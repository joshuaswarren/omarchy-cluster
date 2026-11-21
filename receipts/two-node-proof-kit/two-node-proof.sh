#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
fail() { printf 'two-node-proof: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'EOF'
Usage (fresh OUT and disposable TLS directories; run once on each host):
  two-node-proof.sh mac   SOURCE GGUF OUT LLAMA_SERVER TLS M2_IP RPC_SERVER SPARK_IP
  two-node-proof.sh spark SOURCE GGUF OUT LLAMA_SERVER TLS M2_IP
  two-node-proof.sh --check mac   SOURCE GGUF OUT LLAMA_SERVER TLS M2_IP RPC_SERVER SPARK_IP
  two-node-proof.sh --check spark SOURCE GGUF OUT LLAMA_SERVER TLS M2_IP
SOURCE: git checkout containing commit 10a47b69df53f60b1e9b3f67aa5092fdff2c4562
        (git clone https://github.com/joshuaswarren/omarchy-cluster; git fetch origin 10a47b69df53f60b1e9b3f67aa5092fdff2c4562).
GGUF: exact Qwen3-1.7B-Q4_K_M.gguf, sha256 d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5.
LLAMA_SERVER and RPC_SERVER: llama.cpp builds from the SAME upstream commit on both hosts,
        built with GGML_RPC=ON (Mac also GGML_VULKAN=ON, Spark also GGML_CUDA=ON). The kit records
        each host's llama-server version; the verifier fails when the two differ.
TLS: generate a NEW dedicated CA for each pair per the secure-RPC recipe (cluster-m2-rpc and
        cluster-spark-client). Supply ca.crt and only the local role's key and cert; removed after preflight.
        The TLS directory must be mode 700 and the private key mode 600.
On Mac wait for READY; then start Spark. After Spark finishes, signal Mac with Ctrl-C to retire early.
Mac requires an exclusive GPU window, Vulkan0, AGX busy counters, and sudo -n UFW. Spark needs CUDA0
and per-process nvidia-smi pmon. Both require synchronized clocks (timedatectl), Bash, Python with venv,
pip, system setuptools/wheel, socat (or set SOCAT), OpenSSL, curl, jq, ss, setsid, Git, tar, and pgrep.
Reserve ports 24820, 24831-24832 and 24860-24864; do not change UFW concurrently.
--check runs every prerequisite check and prints one PASS/FAIL line per item; it changes nothing.
EOF
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
check_only=0
if [[ ${1:-} == --check ]]; then check_only=1; shift; fi
case ${1:-}:$# in mac:9|spark:7) ;; *) usage >&2; exit 2 ;; esac
mode=$1
# Portable: GNU realpath -m is missing on macOS, and --check must survive missing paths.
abspath() { if [[ $1 == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi; }
src=$(abspath "$2")
model=$(abspath "$3")
out=$(abspath "$4")
llama=$(abspath "$5")
tls=$(abspath "$6")
m2=$7
peer=$mode
if [[ $mode == mac ]]; then peer=m2; rpc=$(abspath "$8"); spark=$9; fi
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
commit=10a47b69df53f60b1e9b3f67aa5092fdff2c4562
socat=$(command -v "${SOCAT:-socat}" || true)

run_checks() {
    local fails=0
    say() { if [[ $2 == ok ]]; then printf 'PASS %s\n' "$1"; else printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); fi; }
    local label
    if [[ -d $src ]] && git -C "$src" cat-file -e "$commit" 2>/dev/null; then say 'source commit present' ok; else say 'source commit present' no; fi
    if [[ -f $model ]]; then say 'model file present' ok; else say 'model file present' no; fi
    if [[ -f $model ]] && printf '%s  %s\n' d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5 "$model" | sha256sum -c --status; then
        say 'model sha256 matches' ok
    else
        say 'model sha256 matches' no
    fi
    if [[ ! -e $out ]]; then say 'output directory is fresh' ok; else say 'output directory is fresh' no; fi
    if [[ -x $llama ]]; then say 'llama-server executable' ok; else say 'llama-server executable' no; fi
    local probe
    probe=$(mktemp -d)
    # llama-server prints its version on stderr; check-proof.py owns the parser.
    if [[ -x $llama ]] && { "$llama" --version > "$probe/version" 2>&1 || :; } &&
        python3 "$base/check-proof.py" llama-build "$probe/version" >/dev/null 2>&1; then
        say 'llama-server version readable' ok
    else
        say 'llama-server version readable' no
    fi
    if [[ -n $socat ]] && "$socat" -V 2>&1 | grep -q WITH_OPENSSL; then say 'socat with OpenSSL' ok; else say 'socat with OpenSSL' no; fi
    local tool missing=''
    for tool in curl jq ss setsid pgrep git tar openssl python3; do command -v "$tool" >/dev/null || missing="$missing $tool"; done
    if [[ -z $missing ]]; then say 'required tools present' ok; else say "required tools present (missing:$missing)" no; fi
    if python3 -m venv --system-site-packages "$probe/venv" >/dev/null 2>&1 && "$probe/venv/bin/python" -m pip --version >/dev/null 2>&1; then
        say 'python venv with pip' ok
    else
        say 'python venv with pip' no
    fi
    rm -rf "$probe"
    if [[ $(timedatectl show -p NTPSynchronized --value 2>/dev/null) == yes ]]; then say 'clock synchronized' ok; else say 'clock synchronized' no; fi
    local busy=''
    if command -v ss >/dev/null; then
        for p in "${ports[@]}"; do [[ -z $(ss -H -ltn "sport = :$p" 2>/dev/null) ]] || busy="$busy $p"; done
        if [[ -z $busy ]]; then say 'ports free' ok; else say "ports free (in use:$busy)" no; fi
    else
        say 'ports free (ss missing)' no
    fi
    local tls_ok=ok
    [[ -d $tls && $(stat -c %a "$tls" 2>/dev/null) == 700 ]] || tls_ok=no
    [[ -f $tls/$peer.key && $(stat -c %a "$tls/$peer.key" 2>/dev/null) == 600 ]] || tls_ok=no
    for f in ca.crt "$peer.crt" "$peer.key"; do [[ -f $tls/$f && ! -L $tls/$f ]] || tls_ok=no; done
    say 'TLS material and modes' "$tls_ok"
    if python3 -c 'import ipaddress, sys; sys.exit(0 if all(not (a.is_loopback or a.is_unspecified or a.is_multicast) for a in map(ipaddress.IPv4Address, sys.argv[1:])) else 1)' "$m2" "${spark:-$m2}"; then
        say 'peer addresses routable' ok
    else
        say 'peer addresses routable' no
    fi
    if sudo -n true 2>/dev/null; then say 'passwordless sudo' ok; else say 'passwordless sudo' no; fi
    local avail_kib
    avail_kib=$(df -Pk "$(dirname "$out")" 2>/dev/null | awk 'NR==2 {print $4}') || avail_kib=0
    if [[ ${avail_kib:-0} -ge 2097152 ]]; then say 'disk space' ok; else say "disk space (${avail_kib:-0} KiB free)" no; fi
    if [[ $mode == mac ]]; then
        if [[ -r /sys/class/drm/renderD128/device/agx_stats ]]; then say 'AGX busy counters readable' ok; else say 'AGX busy counters readable' no; fi
        if [[ -x ${rpc:-} ]]; then say 'RPC binary executable' ok; else say 'RPC binary executable' no; fi
        if sudo -n ufw status 2>/dev/null | grep -q '^Status: active$'; then say 'UFW active' ok; else say 'UFW active' no; fi
    else
        if command -v nvidia-smi >/dev/null; then say 'nvidia-smi present' ok; else say 'nvidia-smi present' no; fi
        local free_kib
        free_kib=$(awk '$1 == "MemFree:" && $2 ~ /^[0-9]+$/ {print $2}' /proc/meminfo 2>/dev/null) || free_kib=0
        if [[ ${free_kib:-0} -ge 6291456 ]]; then say 'free memory above 6 GiB' ok; else say "free memory above 6 GiB (${free_kib:-0} KiB)" no; fi
    fi
    printf 'CHECKS %s\n' "$fails"
    if [[ $fails -eq 0 ]]; then return 0; fi
    return 2
}

if [[ $check_only == 1 ]]; then
    if [[ $mode == mac ]]; then ports=(24820 24831 24832 24860 24861); else ports=(24820 24831 24832 24864); fi
    run_checks
    exit $?
fi

python3 -c 'import ipaddress, sys; sys.exit(0 if all(not (a.is_loopback or a.is_unspecified or a.is_multicast) for a in map(ipaddress.IPv4Address, sys.argv[1:])) else "expected non-loopback IPv4 peer addresses")' "$m2" "${spark:-$m2}"
[[ -x $llama && -d $src && -f $model && ! -e $out && -f $base/check-proof.py ]] || fail 'invalid binary, input, or output path'
[[ $(stat -c %a "$tls") == 700 && $(stat -c %a "$tls/$peer.key") == 600 ]] || fail 'TLS directory and private key must have mode 700 and 600'
for f in ca.crt "$peer.crt" "$peer.key"; do [[ -f $tls/$f && ! -L $tls/$f ]] || fail 'invalid TLS input'; done
if [[ $mode == mac ]]; then
    openssl verify -CAfile "$tls/ca.crt" -purpose sslserver -verify_hostname cluster-m2-rpc "$tls/m2.crt"
    openssl verify -CAfile "$tls/ca.crt" -purpose sslserver -verify_ip "$m2" "$tls/m2.crt"
else
    openssl verify -CAfile "$tls/ca.crt" -purpose sslclient -verify_hostname cluster-spark-client "$tls/spark.crt"
fi
[[ $(timedatectl show -p NTPSynchronized --value) == yes ]] || fail 'clock is not NTP synchronized'
printf '%s  %s\n' d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5 "$model" 14c3f0ad3a694f784aae16db3cbf845c7d9b6c608904243d7a64498e35e786e2 "$base/request.json" | sha256sum -c --strict
if [[ $mode == mac ]]; then
    ports=(24820 24831 24832 24860 24861)
    stats=/sys/class/drm/renderD128/device/agx_stats
    [[ -x $rpc && -r $stats ]] || fail 'RPC binary or AGX counters not found'
else
    ports=(24820 24831 24832 24864)
    command -v nvidia-smi >/dev/null
fi
for p in "${ports[@]}"; do [[ -z $(ss -H -ltn "sport = :$p") ]] || fail "port $p is in use"; done
mkdir -- "$out"
"$llama" --version > "$out/llama-version.txt" 2>&1 || true
python3 "$base/check-proof.py" llama-build "$out/llama-version.txt" > "$out/llama-build.json" || fail 'cannot read the llama-server version and commit'
mkdir -- "$out/source" "$out/home"
export HOME=$out/home
unset XDG_STATE_HOME XDG_DATA_HOME XDG_CONFIG_HOME XDG_CACHE_HOME
state=$HOME/.local/state/omarchy-cluster/serve.json
log=$HOME/.local/share/omarchy-cluster/llamacpp-engine.log
oc=$out/venv/bin/omarchy-cluster
owns_rule() {
    local current
    current=$(sudo -n ufw status numbered) || return 2
    if [[ $(printf '%s\n' "$current" | grep -F '24861/tcp' | grep -Fw "$spark" | grep -Fc "# $label") == 1 ]]; then return 0; fi
    return 1
}
cleanup() {
    local rc=$? pid own sockets p
    trap '' INT TERM
    trap - EXIT
    if [[ -f $state ]]; then "$oc" stop > "$out/cleanup-stop.log" 2>&1 || rc=75; fi
    if [[ -n ${relay:-} ]]; then kill -TERM -- "-$relay" 2>/dev/null || :; wait "$relay" 2>/dev/null || :; fi
    for pid in ${sampler:-} ${rpc_pid:-}; do kill "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :; done
    if [[ -n ${label:-} ]]; then
        own=0; owns_rule || own=$?
        if [[ $own == 0 ]]; then sudo -n ufw --force delete "${rule[@]}" > "$out/ufw-delete.txt" 2>&1 || rc=75; fi
        [[ $own != 2 ]] || rc=75
        sudo -n ufw status numbered > "$out/ufw-after.txt" || rc=75
        cmp -s "$out/ufw-before.txt" "$out/ufw-after.txt" || rc=75
    fi
    for p in "${ports[@]}"; do
        sockets=$(ss -H -ltn "sport = :$p") && [[ -z $sockets ]] || rc=75
    done
    if [[ $mode == mac ]]; then
        sockets=$(ss -H -tn state established 'sport = :24860 or sport = :24861') && [[ -z $sockets ]] || rc=75
    else
        sockets=$(ss -H -tn state established "sport = :24864 or ( dst $m2 and dport = :24861 )") && [[ -z $sockets ]] || rc=75
    fi
    rm -f -- "$tls/ca.crt" "$tls/$peer.crt" "$tls/$peer.key" || rc=75
    printf '%s\n' "$rc" > "$out/exit-code"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'if [[ ${ready:-} == 1 ]]; then exit 0; else exit 130; fi' INT

openssl x509 -in "$tls/ca.crt" -outform DER | sha256sum | cut -d' ' -f1 > "$out/run-ca.sha256"

git -C "$src" archive "$commit" | tar -x -C "$out/source"
printf '%s\n' "$commit" > "$out/source-commit.txt"
python3 -m venv --system-site-packages "$out/venv"
"$out/venv/bin/python" -m pip install --no-index --no-deps --no-build-isolation "$out/source" > "$out/install.log" 2>&1
"$oc" --version | tee "$out/version.txt" | grep -Fx '0.3.5'
sha256sum "$llama" | cut -d' ' -f1 > "$out/server-sha256.txt"

listen() {
    local pid=$1 p=$2 addr=$3
    for _ in {1..40}; do
        kill -0 "$pid" || return 1
        if ss -H -ltn "sport = :$p" | awk -v addr="$addr:$p" '$4==addr {ok=1} END {exit !(NR==1 && ok)}'; then return 0; fi
        sleep .25
    done
    return 1
}
arm() {
    local name=$1 before engine expected rc free_kib
    shift
    if [[ $mode == spark ]]; then
        nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
            > "$out/$name-resident-mib.txt" 2> "$out/$name-resident.stderr" || :
        sync
        sudo -n sysctl -q vm.drop_caches=3 || fail 'cannot drop page caches'
        grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo > "$out/$name-meminfo-before-engine.txt"
        printf 'Spark %s before engine:\n' "$name"
        cat "$out/$name-meminfo-before-engine.txt"
        free_kib=$(awk '$1 == "MemFree:" && $2 ~ /^[0-9]+$/ && $3 == "kB" {print $2}' "$out/$name-meminfo-before-engine.txt")
        [[ $free_kib =~ ^[0-9]+$ ]] || fail 'cannot read MemFree'
        (( free_kib >= 6 * 1024 * 1024 )) || fail "Spark MemFree ${free_kib} KiB is below the 6 GiB floor after vm.drop_caches=3"
    fi
    "$oc" serve "$model" --engine llamacpp --llama-server "$llama" --ctx 512 --port 24820 --engine-port 24831 \
        --max-concurrent 1 --llama-arg=--gpu-layers --llama-arg=999 --llama-arg=--parallel --llama-arg=1 \
        --llama-arg=--no-warmup --llama-arg=--verbose "$@" > "$out/$name-serve.log" 2>&1 || {
        rc=$?
        cat "$out/$name-serve.log" >&2
        if [[ $mode == spark && -f $log ]]; then
            cp -- "$log" "$out/$name-engine.log"
            grep -E 'CUDA error|cudaSetDevice' "$out/$name-engine.log" >&2 || :
        fi
        return "$rc"
    }
    if [[ $mode == spark ]]; then
        engine=$(jq -er '.local_pids | last' "$state")
        pgrep -P "$engine" -x llama-server > "$out/$name.pid"
        [[ $(wc -l < "$out/$name.pid") == 1 ]]
        expected=CUDA0
    else
        expected=Vulkan0
    fi
    grep -Fq "using device $expected" "$log"
    if [[ $name == shared ]]; then
        grep -Fq 'using device RPC0' "$log"
        cp -- "$log" "$out/shared-engine.log"
        ss -tin state established 'dst 127.0.0.1 and dport = :24864' > "$out/shared-rpc-before.txt"
    fi
    before=$(stat -c %s "$log")
    date -u +%FT%T.%NZ > "$out/$name-start.utc"
    curl --fail-with-body --silent --show-error --max-time 90 -H 'Content-Type: application/json' \
        --data-binary "@$base/request.json" http://127.0.0.1:24820/v1/completions > "$out/$name-response.json"
    date -u +%FT%T.%NZ > "$out/$name-end.utc"
    if [[ $name == shared ]]; then ss -tin state established 'dst 127.0.0.1 and dport = :24864' > "$out/shared-rpc-after.txt"; fi
    tail -c "+$((before+1))" "$log" > "$out/$name-tokens.log"
    python3 "$base/check-proof.py" arm "$out" "$name" | tee "$out/$name-parity.json"
    "$oc" stop > "$out/$name-stop.log" 2>&1
    for p in 24820 24831 24832; do
        for _ in {1..30}; do [[ -n $(ss -H -ltn "sport = :$p") ]] || break; sleep .2; done
        [[ -z $(ss -H -ltn "sport = :$p") ]]
    done
}
if [[ $mode == mac ]]; then
    (
        while :; do
            printf '%s %s\n' "$(date -u +%FT%T.%NZ)" "$(awk '/^busy_ns |^jobs / {print $2}' "$stats" | paste -sd ' ')"
            sleep .2
        done
    ) > "$out/gpu.tsv" & sampler=$!
    arm mac --llama-arg=--device --llama-arg=Vulkan0
    sudo -n ufw status numbered > "$out/ufw-before.txt"
    grep -q '^Status: active$' "$out/ufw-before.txt" || fail 'UFW inactive'
    if grep -qw 24861 "$out/ufw-before.txt"; then fail 'existing rule for 24861'; fi
    "$rpc" -d Vulkan0 -H 127.0.0.1 -p 24860 > "$out/rpc.log" 2>&1 & rpc_pid=$!
    listen "$rpc_pid" 24860 127.0.0.1
    setsid "$socat" -d -d "OPENSSL-LISTEN:24861,bind=$m2,range=$spark/32,reuseaddr,fork,verify=1,cert=$tls/m2.crt,key=$tls/m2.key,cafile=$tls/ca.crt,commonname=cluster-spark-client" \
        TCP4:127.0.0.1:24860 > "$out/tls.log" 2>&1 & relay=$!
    listen "$relay" 24861 "$m2"
    label="cluster-proof-$(openssl rand -hex 8)"
    rule=(allow from "$spark" to any port 24861 proto tcp)
    sudo -n ufw "${rule[@]}" comment "$label" > "$out/ufw-add.txt"
    owns_rule
    sudo -n ufw status numbered > "$out/ufw-during.txt"
    date -u +%FT%T.%NZ | tee "$out/ready"
    printf 'READY: start the Spark command now.\n'
    ready=1
    end=$((SECONDS+240))
    while (( SECONDS < end )) && [[ ! -e $out/finish-request ]]; do kill -0 "$rpc_pid" "$relay"; sleep .5; done
else
    TZ=UTC nvidia-smi -i 0 --query-compute-apps=timestamp,pid,used_memory \
        --format=csv,noheader,nounits --loop=1 > "$out/gpu-mem.csv" & sampler=$!
    arm spark --llama-arg=--device --llama-arg=CUDA0
    if timeout 6 openssl s_client -brief -tls1_2 -connect "$m2:24861" -verify_ip "$m2" -verify_return_error \
        -CAfile "$tls/ca.crt" </dev/null > "$out/tls-no-client.log" 2>&1; then fail 'TLS accepted a missing certificate'; fi
    grep -Eq 'alert (handshake failure|certificate required)|peer did not return a certificate' "$out/tls-no-client.log"
    timeout 6 openssl s_client -brief -connect "$m2:24861" -verify_ip "$m2" -verify_return_error \
        -CAfile "$tls/ca.crt" -cert "$tls/spark.crt" -key "$tls/spark.key" </dev/null > "$out/tls-valid-peer.log" 2>&1
    grep -Fq 'Verification: OK' "$out/tls-valid-peer.log"
    setsid "$socat" -d -d TCP4-LISTEN:24864,bind=127.0.0.1,reuseaddr,fork \
        "OPENSSL:$m2:24861,verify=1,cert=$tls/spark.crt,key=$tls/spark.key,cafile=$tls/ca.crt,commonname=cluster-m2-rpc" \
        > "$out/tls.log" 2>&1 & relay=$!
    listen "$relay" 24864 127.0.0.1
    arm shared --rpc-node 127.0.0.1:24864=2 --host-layers 0 --llama-arg=--device --llama-arg=CUDA0,RPC0 \
        --llama-arg=--tensor-split --llama-arg=14,15
    kill -0 "$sampler"
    printf 'Spark complete. Stop the Mac proof after collecting both output directories.\n'
fi
