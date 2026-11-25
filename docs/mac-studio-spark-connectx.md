# Set up direct Ethernet between a Mac Studio and a DGX Spark

With a ConnectX-5 Ex in a Thunderbolt dock, we got a direct TCP link from an M1 Ultra Mac Studio to a DGX Spark-class host. Both ends linked at 100 Gb/s full duplex. Our best 10-second runs reached 12.30 Gb/s to the Spark and 15.27 Gb/s to the Mac.

## Hardware and port

| Part | Tested setup |
| --- | --- |
| Mac | M1 Ultra Mac Studio, macOS 26.6.2 (25G83) |
| Dock | Powered PCIe dock listed as `TB4 eGFX`, firmware 33.1. It negotiated Thunderbolt 3 at 40 Gb/s. |
| Mac NIC | Mellanox ConnectX-5 Ex (`15b3:1019`), PCIe Gen3 x4 |
| Linux peer | Acer Veriton GN100 (DGX Spark class), built-in ConnectX-7 |
| Ethernet cable | QSFP28 direct-attach copper, 100GBASE-CR4 |

macOS found both PCI functions with the dock on a rear Thunderbolt port. The built-in `AppleEthernetMLX5` driver launched, then quit about four seconds later. No network interface showed up. When we moved just the dock's Thunderbolt cable to the front port (bus 5, receptacle 6 on this Mac), the driver bound. We did not find the cause of the first failure or change any security settings.

See [Apple's port specs](https://support.apple.com/en-us/111900) before you move the cable. The 2022 M1 Ultra has front Thunderbolt ports. The M1 Max's front USB-C ports do not support Thunderbolt.

On the Mac, check the dock, PCI functions and interfaces:

```sh
system_profiler SPThunderboltDataType SPPCIDataType
ifconfig -l
ifconfig en24
networksetup -listvalidMTUrange en24
```

Confirm `Driver Installed: Yes`, then match the active interface to the card. We got `en23` and `en24`, but only `en24` showed an active `100GBase-CR4` link. On the Spark, use `ip -br link` and `ethtool` to find the cabled port. Two ConnectX-7 interfaces may show carrier for one physical cage. Pick one.

## Set up the direct link

First make sure neither host uses `10.77.0.0/30`. Keep your management link. Use only idle test ports with no configured IP address or default route. Use your own device names if they differ. Do not set a gateway or DNS server on this link.

On the Mac, check the interface and routes first:

```sh
MAC_IF=en24
ifconfig "$MAC_IF"
netstat -rn -f inet
```

If the port and subnet are free, add the alias:

```sh
sudo ifconfig "$MAC_IF" inet 10.77.0.1 netmask 255.255.255.252 alias
ifconfig "$MAC_IF"
route -n get 10.77.0.2
```

On the Spark, check the port, routes and NetworkManager profiles. Pick a profile name that is not in the list. If the test port has an active profile, note it so you can restore it later.

```sh
SPARK_IF=enp1s0f0np0
PROFILE=connectx-direct-test
sudo ethtool "$SPARK_IF"
ip -br address show dev "$SPARK_IF"
ip -4 route show
nmcli -f NAME,DEVICE connection show
```

If the port, subnet and name checks pass, add and start a temporary profile:

```sh
sudo nmcli connection add save no type ethernet ifname "$SPARK_IF" con-name "$PROFILE" \
  connection.autoconnect no ipv4.method manual ipv4.addresses 10.77.0.2/30 \
  ipv4.never-default yes ipv6.method link-local ipv6.never-default yes \
  802-3-ethernet.mtu 1500 802-3-ethernet.cloned-mac-address preserve &&
  sudo nmcli connection up id "$PROFILE" ifname "$SPARK_IF"
ip -br address show dev "$SPARK_IF"
ip -4 route get 10.77.0.1
```

Check that both interfaces show MTU 1500. The Mac listed 1280 to 2034 bytes and would not accept 9000. Check both peer routes before you run a test. Each should use its dedicated interface.

## Run the TCP tests

Use `iperf3` on both hosts. We used 3.22.1 on the Mac and 3.16 on the Spark. From the Spark, ping the Mac. Bind the server to the test address and leave it in the foreground so you can stop it later.

```sh
ping -c 3 -I 10.77.0.2 10.77.0.1
iperf3 -s -B 10.77.0.2
```

On the Mac, ping the Spark and test both ways. Use `-R` when the Spark sends.

```sh
ping -c 3 -S 10.77.0.1 10.77.0.2
iperf3 -c 10.77.0.2 -B 10.77.0.1 -P 8 -t 10
iperf3 -c 10.77.0.2 -B 10.77.0.1 -P 8 -t 10 -R
```

To repeat the other two runs in each direction, use `-P 16` in place of `-P 8`. Run once with default socket buffers and once with `-w 4M`.

| Streams / requested socket buffer | Mac to Spark (Gb/s) | Retransmits | Spark to Mac (Gb/s) | Retransmits |
| --- | ---: | ---: | ---: | ---: |
| 8 / default | 12.30 | 0 | 15.27 | 13,817 |
| 16 / default | 11.80 | 24,764 | 12.51 | 17,034 |
| 16 / 4 MiB | 12.15 | 28,581 | 13.53 | 6,982 |

Each rate comes from the receiver. Each retransmit count comes from the sender of that run. All three pings came back in each direction. The 100 Gb/s Ethernet link rate is not the TCP rate through the Thunderbolt tunnel. The fastest run to the Mac had 13,817 retransmits. We measured TCP, not RDMA or distributed inference.

## Why this test used TCP

[Apple's RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt) links Thunderbolt 5 Macs. It does not provide RoCE on this card.

We checked the M1 Ultra's running arm64e boot image and the `kernelcache.release.mac13j` image in [Apple's 25G83 restore](https://updates.cdn-apple.com/2026SummerFCS/fullrestores/140-75212/A2A24B94-1FC1-45A3-93F7-C51B02AF1F4D/UniversalMac_26.6.2_25G83_Restore.ipsw). Neither contained `IORDMAFamily`. The M3 Ultra image from the same restore, `kernelcache.release.mac15j`, did. Apple's `26A428` restore still omits the family on the M1 Ultra.

The [MCDMA kext](https://github.com/ashhart/MCDMA/blob/c07ca4e28ed129f56c26ae25233b5063495b59f3/docs/install.md) needs that kernel family. An allowlist change will not make it work on those M1 Ultra builds. A DriverKit driver with its own verbs library can work without that family.

## Remove the test settings

Stop the server with Ctrl-C. The addresses do not survive a reboot. To remove them now, undo only the alias and profile you added:

- Mac: `sudo ifconfig "$MAC_IF" inet 10.77.0.1 -alias`
- Spark: `sudo nmcli connection delete id "$PROFILE"`. Restore the previous profile if it did not reconnect on its own.
