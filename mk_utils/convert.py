import os
import subprocess

def run_conversion_script():
    script_path = os.path.join('doc', 'convert_to_md.py')
    if os.path.exists(script_path):
        print(f"Running conversion script: {script_path}")
        subprocess.run(['python', script_path], check=True)
    else:
        print(f"Conversion script not found at: {script_path}")
