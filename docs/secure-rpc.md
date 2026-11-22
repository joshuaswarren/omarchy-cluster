# Authenticate a cross-host llama.cpp RPC link

`ggml-rpc-server` does not authenticate its clients. Keep raw RPC on loopback on both machines. This recipe uses mutual TLS outside omarchy-cluster: a Linux Mac serves a Vulkan device, and a DGX Spark runs the gateway and CUDA device. Only the TLS listener crosses the LAN. This does not secure the gateway's HTTP API or SSH.

Use compatible RPC-enabled llama.cpp builds, `socat` with OpenSSL support on both hosts, OpenSSL, and an active UFW firewall on the Mac. Reserve TCP ports 24860–24861 on the Mac and 24864 on Spark for this link. If a listener or firewall rule already uses those ports, stop and choose unused ports instead. The commands below use Bash. Replace the documentation-only IPv4 addresses with the two LAN addresses and set these variables in each terminal:

```bash
M2_ADDR=192.0.2.10
SPARK_ADDR=192.0.2.20
```

## Make short-lived certificates

On a trusted staging host, create a dedicated CA and two separate leaf keys. The Mac certificate includes the Mac IP and is for server authentication only; Spark's is for client authentication only.

```bash
umask 077
stage=$(mktemp -d) || exit 1
cd "$stage" || exit 1
mkdir m2 spark || exit 1
openssl req -new -newkey rsa:2048 -nodes -x509 -sha256 -days 1 \
  -subj /CN=cluster-rpc-ca -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' -keyout ca.key -out ca.crt
for peer in m2 spark; do
  case $peer in
    m2) cn=cluster-m2-rpc; purpose=serverAuth; san=DNS:cluster-m2-rpc,IP:$M2_ADDR ;;
    spark) cn=cluster-spark-client; purpose=clientAuth; san=DNS:$cn ;;
  esac
  openssl req -new -newkey rsa:2048 -nodes -subj "/CN=$cn" \
    -keyout "$peer/$peer.key" -out "$peer/$peer.csr"
  printf 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=%s\nsubjectAltName=%s\n' "$purpose" "$san" |
    openssl x509 -req -sha256 -days 1 -CA ca.crt -CAkey ca.key -CAcreateserial \
      -extfile /dev/stdin -in "$peer/$peer.csr" -out "$peer/$peer.crt"
  cp ca.crt "$peer/ca.crt"
done
openssl verify -CAfile ca.crt -purpose sslserver -verify_ip "$M2_ADDR" m2/m2.crt
openssl verify -CAfile ca.crt -purpose sslserver -verify_hostname cluster-m2-rpc m2/m2.crt
openssl verify -CAfile ca.crt -purpose sslclient spark/spark.crt
rm -- ca.key m2/m2.csr spark/spark.csr
```

Over an already trusted channel (such as SSH with host-key checking), copy `m2/{ca.crt,m2.crt,m2.key}` to a private directory on the Mac and `spark/{ca.crt,spark.crt,spark.key}` to a private directory on Spark. Keep each directory mode `0700` and each private key mode `0600`. Set `TLS` to the appropriate absolute directory in each terminal. Do not copy the other host's private key or install this CA system-wide.

## Open only the authenticated link

On the Mac, set `RPC_BIN` to your `ggml-rpc-server` binary. Do not run this alongside other firewall changes. Before starting, check that UFW is active and has no existing rule for this listener, including ranges or application profiles:

```bash
sudo ufw status numbered > ufw-before.txt || exit 1
cat ufw-before.txt
grep -q '^Status: active$' ufw-before.txt || exit 1
if grep -qw 24861 ufw-before.txt; then exit 1; fi
rule_tag="omarchy-cluster RPC mTLS $(openssl rand -hex 8)"
"$RPC_BIN" -d Vulkan0 -H 127.0.0.1 -p 24860 & rpc_pid=$!
socat -d -d \
  "OPENSSL-LISTEN:24861,bind=$M2_ADDR,range=$SPARK_ADDR/32,reuseaddr,fork,verify=1,cert=$TLS/m2.crt,key=$TLS/m2.key,cafile=$TLS/ca.crt,commonname=cluster-spark-client" \
  TCP4:127.0.0.1:24860 & mtls_pid=$!
ss -H -ltn 'sport = :24860' | awk '$4=="127.0.0.1:24860" {ok=1} END {exit !(NR==1 && ok)}'
ss -H -ltn 'sport = :24861' | awk -v addr="$M2_ADDR:24861" '$4==addr {ok=1} END {exit !(NR==1 && ok)}'
```

Wait until both socket checks succeed. If either process exits or binds to a different address, stop it and do not open the firewall. Otherwise add one rule from Spark's IP:

```bash
sudo ufw allow from "$SPARK_ADDR" to any port 24861 proto tcp comment "$rule_tag"
sudo ufw status numbered | tee ufw-during.txt
```

From Spark, verify that the first command is rejected **for missing a client certificate** (a timeout is not a passing check). The second must report `Verification: OK` and exit successfully:

```bash
timeout 6 openssl s_client -brief -tls1_2 -connect "$M2_ADDR:24861" \
  -verify_ip "$M2_ADDR" -CAfile "$TLS/ca.crt" -verify_return_error </dev/null
timeout 6 openssl s_client -brief -connect "$M2_ADDR:24861" \
  -verify_ip "$M2_ADDR" -CAfile "$TLS/ca.crt" -cert "$TLS/spark.crt" \
  -key "$TLS/spark.key" -verify_return_error </dev/null
```

On Spark, start the loopback client relay. Give `omarchy-cluster serve … --engine llamacpp` the relay address as `--rpc-node 127.0.0.1:24864=2` (adjust the remote memory budget for your run). Do not point it at the Mac's raw RPC port.

```bash
socat -d -d TCP4-LISTEN:24864,bind=127.0.0.1,reuseaddr,fork \
  "OPENSSL:$M2_ADDR:24861,verify=1,cert=$TLS/spark.crt,key=$TLS/spark.key,cafile=$TLS/ca.crt,commonname=cluster-m2-rpc" & relay_pid=$!
ss -H -ltn 'sport = :24864' | awk '$4=="127.0.0.1:24864" {ok=1} END {exit !(NR==1 && ok)}'
```

Wait for that socket check to succeed. Use a fresh private `RUN_HOME` for the test gateway, rather than sharing state with an existing instance. Set `GGUF` and `LLAMA_SERVER` to absolute paths and send a valid completion body (`request.json`):

```bash
RUN_HOME=$(mktemp -d) || exit 1
HOME="$RUN_HOME" omarchy-cluster serve "$GGUF" --engine llamacpp \
  --llama-server "$LLAMA_SERVER" --ctx 512 --port 24820 --engine-port 24831 \
  --rpc-node 127.0.0.1:24864=2 --host-layers 0 \
  --llama-arg=--device --llama-arg=CUDA0,RPC0 --llama-arg=--tensor-split --llama-arg=14,15 \
  --llama-arg=--gpu-layers --llama-arg=999
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @request.json http://127.0.0.1:24820/v1/completions
```

The completion must succeed through the relay, and the engine log in `RUN_HOME` must show both `using device CUDA0` and `using device RPC0`. A direct TLS handshake alone does not verify RPC through the relay.

## Tear down

On Spark, stop the test gateway first with `HOME="$RUN_HOME" omarchy-cluster stop`. Close both relays and the worker using their captured PIDs in the same shells. On the Mac, check the live rule still matches your unique comment and tuple before deleting it:

```bash
kill "$mtls_pid" "$rpc_pid"
wait "$mtls_pid" "$rpc_pid" 2>/dev/null || true
current=$(sudo ufw status numbered) || exit 1
if printf '%s\n' "$current" | grep -F '24861/tcp' | grep -F "$SPARK_ADDR" | grep -Fq "# $rule_tag"; then
  sudo ufw --force delete allow from "$SPARK_ADDR" to any port 24861 proto tcp
else
  printf '%s\n' 'Owned rule not found; inspect UFW before deleting anything.' >&2
fi
sudo ufw status numbered | tee ufw-after.txt
cmp ufw-before.txt ufw-after.txt
ss -H -ltn 'sport = :24860 or sport = :24861'
ss -H -tn state established 'sport = :24860 or sport = :24861'
rm -- "$TLS/m2.key" "$TLS/m2.crt" "$TLS/ca.crt"
```

On Spark, stop the local relay and remove its credentials:

```bash
kill "$relay_pid"
wait "$relay_pid" 2>/dev/null || true
ss -H -ltn 'sport = :24864'
ss -H -tn state established 'sport = :24864 or dport = :24861'
rm -- "$TLS/spark.key" "$TLS/spark.crt" "$TLS/ca.crt"
```

The `ss` checks after teardown must print nothing. Remove the remaining staging copies of the test keys and certificates. If setup aborts after opening the firewall, still remove the scoped rule and verify the before/after states. This setup does not implement failover: if the RPC worker disappears, in-flight or subsequent requests can fail. Restart the worker and the test gateway before retrying.
