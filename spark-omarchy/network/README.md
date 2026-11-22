# Network

- `avahi` + `nss-mdns`: publish the node as `<hostname>.local` on the LAN so
  the cluster planner can find it without a fixed address. Enable
  `avahi-daemon.service` and `avahi-daemon.socket`.
- `ufw`: apply `ufw.commands` with the `ufw` commands shown there (comments
  included). Default-deny inbound, allow ssh + mDNS + the Remnic ports scoped
  to the LAN prefix. The RPC port stays closed between runs.
- The kit does not open any inbound port for llama.cpp RPC. The proof kit
  manages one temporary, labelled, peer-scoped rule itself.
