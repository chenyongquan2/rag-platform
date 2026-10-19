"""Print the machine load once: CPU percent over one second and the five busiest processes.

Used by the run scripts so every run in the log has the load it ran under (spec P2-03).
"""

import psutil

cpu = psutil.cpu_percent(interval=1.0)
procs = []
for p in psutil.process_iter(["name", "cpu_percent"]):
    procs.append((p.info["cpu_percent"] or 0.0, p.info["name"]))
procs.sort(reverse=True)
top = ", ".join(f"{name} {pct:.0f}%" for pct, name in procs[:5])
print(f"load: cpu {cpu:.0f}% of {psutil.cpu_count()} logical cores; top: {top}")
