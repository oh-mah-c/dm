import subprocess
import os

env = os.environ.copy()
env["MALLOC_CHECK_"] = "3"

open("gdb_script.txt", "w").write("run resnet18 bench\nbt\nquit\n")
proc = subprocess.run(["gdb", "-batch", "-x", "gdb_script.txt", "bin/dm.exe"], capture_output=True, text=True, env=env)
print(proc.stdout)
print(proc.stderr)
