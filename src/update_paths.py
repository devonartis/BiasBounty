"""
Script to update paths in all source files
"""

import os
import glob
import shutil

def backup_file(filepath):
    """Create backup of original file"""
    backup_path = filepath + '.bak'
    shutil.copy2(filepath, backup_path)
    return backup_path

def update_paths_in_file(filepath):
    """Update paths in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace old paths with new config-based paths
    replacements = {
        "'./Training_data'": "PATHS['TRAIN_DATA']",
        "'./Test_data'": "PATHS['TEST_DATA']",
        "'./Adversarial_data'": "PATHS['PERTURBED']",
        "'./model'": "PATHS['MODEL']",
        "'predictions.csv'": "PREDICTIONS_CSV",
        "'adversarial_predictions.csv'": "ADVERSARIAL_PREDICTIONS_CSV"
    }

    # Add import if needed
    if 'from config import' not in content:
        content = 'from config import PATHS, MODEL_PATH, PREDICTIONS_CSV, ADVERSARIAL_PREDICTIONS_CSV\n' + content

    # Make replacements
    for old, new in replacements.items():
        content = content.replace(old, new)

    # Save updated content
    with open(filepath, 'w') as f:
        f.write(content)

def main():
    """Update all Python files in src directory"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    python_files = glob.glob(os.path.join(src_dir, '*.py'))

    for filepath in python_files:
        if os.path.basename(filepath) not in ['config.py', 'update_paths.py']:
            print(f"Updating {filepath}...")
            backup = backup_file(filepath)
            try:
                update_paths_in_file(filepath)
                print(f"Successfully updated {filepath}")
            except Exception as e:
                print(f"Error updating {filepath}: {e}")
                print(f"Restoring from backup...")
                shutil.copy2(backup, filepath)

    print("\nPath updates complete. Please review the changes.")

if __name__ == "__main__":
    main()