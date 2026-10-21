"""Fixed English taxonomy for the 15 raw Chinese failure_type strings in CCF AIOps 2022.

Confirmed exhaustive across all 7 training groundtruth CSVs (300 events) and all 5 test
groundtruth JSONs (241 events) — same 15 strings in both, no unseen variants.
"""

FAILURE_TAXONOMY = {
    "k8s容器cpu负载": ("pod_cpu_load", "pod"),
    "k8s容器内存负载": ("pod_memory_load", "pod"),
    "k8s容器读io负载": ("pod_io_read_load", "pod"),
    "k8s容器写io负载": ("pod_io_write_load", "pod"),
    "k8s容器网络丢包": ("pod_network_packet_loss", "pod"),
    "k8s容器网络资源包损坏": ("pod_network_packet_corruption", "pod"),
    "k8s容器网络资源包重复发送": ("pod_network_packet_duplication", "pod"),
    "k8s容器网络延迟": ("pod_network_latency", "pod"),
    "k8s容器进程中止": ("pod_process_abort", "pod"),
    "node节点CPU故障": ("node_cpu_fault", "node"),
    "node节点CPU爬升": ("node_cpu_climb", "node"),
    "node 磁盘读IO消耗": ("node_disk_read_io_consumption", "node"),
    "node 磁盘写IO消耗": ("node_disk_write_io_consumption", "node"),
    "node 磁盘空间消耗": ("node_disk_space_consumption", "node"),
    "node 内存消耗": ("node_memory_consumption", "node"),
}


def translate(failure_type_raw: str) -> str:
    if failure_type_raw not in FAILURE_TAXONOMY:
        raise KeyError(
            f"Unseen failure_type not in fixed taxonomy: {failure_type_raw!r} — "
            "extend FAILURE_TAXONOMY rather than guessing a translation."
        )
    return FAILURE_TAXONOMY[failure_type_raw][0]
