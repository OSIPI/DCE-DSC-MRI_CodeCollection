import os
import shutil

#copy files and folders from notebooks directory to docs directory
def copy_assets():
    source_folder = 'notebooks'
    destination_folder = 'docs'
    excluded_dirs = {'old_nb'}
    excluded_extensions = {'.yml'}  # Example extensions to skip
    excluded_filenames = {'requirements.txt'}      # Example specific files to skip {"readme.md","index.md"}
    
    for root, dirs, files in os.walk(source_folder):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        rel_path = os.path.relpath(root, source_folder)
        dest_path = os.path.join(destination_folder, rel_path)
        os.makedirs(dest_path, exist_ok=True)

        for file in files:
            if file in excluded_filenames or os.path.splitext(file)[1] in excluded_extensions:
                continue

            src = os.path.join(root, file)
            dst = os.path.join(dest_path, file)
            shutil.copy2(src, dst)
