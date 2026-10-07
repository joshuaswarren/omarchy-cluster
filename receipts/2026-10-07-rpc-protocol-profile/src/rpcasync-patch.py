"""Patch llama.cpp 65840ed ggml-rpc.cpp: queue small SET_TENSOR writes asynchronously on the
device backend and synchronize once before any other command. Run in the llama.cpp root."""
import sys

p = "ggml/src/ggml-rpc/ggml-rpc.cpp"
s = open(p).read()


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit("patch anchor not found exactly once:\n" + old)
    s = s.replace(old, new)


sub("""        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
    }""", """        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        pending_dirty.resize(backends.size(), false);
    }
    // wait for queued small writes; called before every command that is not SET_TENSOR
    void flush_writes();""")

sub("""    std::vector<ggml_backend_t> backends;
    const char * cache_dir;""", """    int device_of(ggml_backend_buffer_t buffer) const;
    std::vector<ggml_backend_t> backends;
    // small SET_TENSOR payloads written with ggml_backend_tensor_set_async, kept alive until flush_writes
    std::vector<std::vector<uint8_t>> pending_writes;
    std::vector<bool> pending_dirty;
    const char * cache_dir;""")

sub("""    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(""", """    // Inputs, positions and masks arrive as several small writes per token. A synchronous write to a
    // GPU buffer is a staging copy, a submit and a wait each; queue them and wait once instead.
    const int dev = (size <= (1u << 20) && offset + size <= ggml_nbytes(tensor)) ? device_of(tensor->buffer) : -1;
    if (dev >= 0) {
        pending_writes.emplace_back((const uint8_t *) data, (const uint8_t *) data + size);
        ggml_backend_tensor_set_async(backends[dev], tensor, pending_writes.back().data(), offset, size);
        pending_dirty[dev] = true;
    } else {
        flush_writes();
        ggml_backend_tensor_set(tensor, data, offset, size);
    }
    return true;
}
int rpc_server::device_of(ggml_backend_buffer_t buffer) const {
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    for (size_t i = 0; i < backends.size(); i++) {
        if (ggml_backend_get_default_buffer_type(backends[i]) == buft) {
            return (int) i;
        }
    }
    return -1;
}
void rpc_server::flush_writes() {
    for (size_t i = 0; i < backends.size(); i++) {
        if (pending_dirty[i]) {
            ggml_backend_synchronize(backends[i]);
            pending_dirty[i] = false;
        }
    }
    pending_writes.clear();
}

bool rpc_server::get_cached_file(""")

sub("""        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\\n", cmd);
            break;
        }
        switch (cmd) {""", """        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\\n", cmd);
            break;
        }
        if (cmd != RPC_CMD_SET_TENSOR) {
            server.flush_writes();
        }
        switch (cmd) {""")

sub("""rpc_server::~rpc_server() {
    for (auto buffer : buffers) {""", """rpc_server::~rpc_server() {
    flush_writes();  // queued writes may target the buffers freed below
    for (auto buffer : buffers) {""")

open(p, "w").write(s)
print("patched", p)
