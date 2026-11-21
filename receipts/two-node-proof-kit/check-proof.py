import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOKEN = re.compile(r'n_gen = (\d+), n_remaining = -?\d+, next token:\s+(\d+)')
REFERENCE_ID_SHA = '284de5541d2d52e9678ae8f832eb0efcb5d6c72fe605e653b84ddfb4976bf682'
ID_SHA = REFERENCE_ID_SHA  # kept for callers that pin our reference digest


def parse_arm(root, name):
    response = json.loads((root / f'{name}-response.json').read_text())
    usage = response['usage']
    if (usage['prompt_tokens'], usage['completion_tokens']) != (43, 64):
        raise ValueError(f'{name}: expected 43 prompt and 64 output tokens')
    pairs = [(int(index), int(token)) for index, token in TOKEN.findall((root / f'{name}-tokens.log').read_text())]
    if [index for index, _ in pairs] != list(range(1, 65)):
        raise ValueError(f'{name}: incomplete or duplicate decode log')
    ids = [token for _, token in pairs]
    digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    rate = response['timings']['tokens_per_s']
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f'{name}: invalid decode rate')
    # The reference digest is our run, not a requirement: an outside tester's
    # tokens may differ. Parity across the three arms is what the proof needs.
    return {
        'tokens': len(ids),
        'ids_sha256': digest,
        'matches_reference': digest == REFERENCE_ID_SHA,
        'decode_tokens_per_s': rate,
    }


def check_parity(digests):
    if len(set(digests)) != 1:
        raise ValueError(f'token IDs differ between arms: {sorted(set(digests))}')
    return True


def check_llama_versions(mac_text, spark_text):
    def fields(text):
        version = re.search(r'version:\s*(\S+)', text)
        commit = re.search(r'commit:\s*([0-9a-f]+)', text)
        return (version.group(1) if version else None, commit.group(1) if commit else None)

    mac, spark = fields(mac_text), fields(spark_text)
    if mac != spark or mac == (None, None):
        raise ValueError(f'llama-server builds differ between hosts: mac {mac}, spark {spark}')
    return True



def stamp(text):
    return datetime.fromisoformat(text.strip().replace('Z', '+00:00'))


def window(root, name):
    start, end = (stamp((root / f'{name}-{edge}.utc').read_text()) for edge in ('start', 'end'))
    if end <= start:
        raise ValueError(f'{name}: invalid request window')
    return start, end


def agx(path, bounds):
    lo, hi = bounds
    samples = [(stamp(t), int(busy), int(jobs)) for t, busy, jobs in (line.split() for line in path.read_text().splitlines()) if lo <= stamp(t) <= hi]
    if len(samples) < 2:
        raise ValueError('missing Vulkan GPU samples inside request window; check host clocks')
    first, last = samples[0], samples[-1]
    busy, jobs = last[1] - first[1], last[2] - first[2]
    if busy <= 0 or jobs <= 0:
        raise ValueError('no Vulkan GPU work during request')
    return {'busy_ns': busy, 'jobs': jobs, 'samples': len(samples)}


def cuda_model_mib(path):
    log = path.read_text(errors='replace')
    for device in ('CUDA0 (', 'RPC0 (127.0.0.1:24864)'):
        if f'llama_prepare_model_devices: using device {device}' not in log:
            raise ValueError(f'shared: engine did not select {device}')
    if 'load_tensors: offloaded 29/29 layers to GPU' not in log:
        raise ValueError('shared: missing 29/29 layer offload')
    buffers = re.findall(r'(?m)load_tensors:[ \t]+CUDA0 model buffer size =[ \t]+(\d+(?:\.\d+)?) MiB$', log)
    if not buffers or not 0 < float(buffers[-1]) < math.inf:
        raise ValueError('shared: missing nonzero CUDA0 model buffer')
    return float(buffers[-1])


def shared_cuda(root, bounds):
    model = cuda_model_mib(root / 'shared-engine.log')
    pid = (root / 'shared.pid').read_text().strip()
    if not re.fullmatch(r'[1-9][0-9]*', pid):
        raise ValueError('shared: invalid engine PID')
    capture = root / 'gpu-mem.csv'
    if not capture.is_file():
        raise ValueError('shared: missing per-PID GPU memory capture')
    lo, hi = bounds
    memory = []
    for line in capture.read_text().splitlines():
        when, observed_pid, mib = map(str.strip, line.split(','))
        if observed_pid != pid:
            continue
        timestamp = datetime.strptime(when, '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
        if lo <= timestamp <= hi and mib.isdecimal() and int(mib) > 0:
            memory.append(int(mib))
    if not memory:
        raise ValueError('shared: no positive per-PID GPU memory sample in request window')
    return {'model_buffer_mib': model, 'pid_memory_mib': memory}


def rpc_received(path):
    counts = re.findall(r'\bbytes_received:(\d+)\b', path.read_text())
    if len(counts) != 1:
        raise ValueError('expected one active test RPC connection with byte counters')
    return int(counts[0])

def verify(mac, spark):
    if any((root / 'exit-code').read_text().strip() != '0' for root in (mac, spark)):
        raise ValueError('proof or teardown failed on at least one host')
    run_ids = [(root / 'run-ca.sha256').read_text().strip() for root in (mac, spark)]
    if run_ids[0] != run_ids[1] or not re.fullmatch(r'[0-9a-f]{64}', run_ids[0]):
        raise ValueError('run receipts use different test CAs')
    if (mac / 'ufw-before.txt').read_bytes() != (mac / 'ufw-after.txt').read_bytes():
        raise ValueError('firewall not restored')
    single_mac, single_spark, shared = (parse_arm(root, name) for root, name in ((mac, 'mac'), (spark, 'spark'), (spark, 'shared')))
    check_parity([single_mac['ids_sha256'], single_spark['ids_sha256'], shared['ids_sha256']])
    check_llama_versions(
        (mac / 'llama-version.txt').read_text(),
        (spark / 'llama-version.txt').read_text(),
    )
    single_mac['vulkan'] = agx(mac / 'gpu.tsv', window(mac, 'mac'))
    shared_window = window(spark, 'shared')
    shared['vulkan'] = agx(mac / 'gpu.tsv', shared_window)
    shared['cuda'] = shared_cuda(spark, shared_window)
    rpc_delta = rpc_received(spark / 'shared-rpc-after.txt') - rpc_received(spark / 'shared-rpc-before.txt')
    if rpc_delta <= 0:
        raise ValueError('no RPC replies were observed during the shared request')
    shared['rpc_reply_bytes'] = rpc_delta
    return {'exact_token_parity': True, 'firewall_restored': True, 'mac_only': single_mac, 'spark_only': single_spark, 'shared': shared}


def main():
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == 'arm' and args[2] in ('mac', 'spark', 'shared'):
        result = parse_arm(Path(args[1]), args[2])
    elif len(args) == 3 and args[0] == 'combine':
        result = verify(Path(args[1]), Path(args[2]))
    else:
        raise SystemExit('usage: check-proof.py arm RUN {mac|spark|shared} | combine MAC_RUN SPARK_RUN')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
