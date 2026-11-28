import runpy
import sys

sys.path.insert(0, "/home/node/.local/lib/python3.11/site-packages")
sys.path.insert(0, "/workspace")
runpy.run_path("/workspace/tools/sourcemap_demo.py", run_name="__main__")
