#!/usr/bin/env python3
"""
Script to fix BSK-RL imports in all Python files
"""
import os
import re

def fix_bsk_rl_imports(file_path):
    """Fix BSK-RL imports in a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file already uses our setup helper
    if 'from bsk_rl_setup import get_bsk_rl_imports' in content:
        print(f"✅ {file_path} already fixed")
        return
    
    # Pattern to match BSK-RL imports
    bsk_import_pattern = r'from bsk_rl import.*\nfrom bsk_rl\.sim import.*\nfrom Basilisk\.architecture import.*'
    
    # Replacement with our setup helper
    replacement = """# Import BSK-RL using setup helper
from bsk_rl_setup import get_bsk_rl_imports
act, data, obs, scene, sats, dyn, fsw, bskLogging = get_bsk_rl_imports()"""
    
    # Try to replace the pattern
    new_content = re.sub(bsk_import_pattern, replacement, content, flags=re.MULTILINE)
    
    # If no change, try individual patterns
    if new_content == content:
        # Try simpler patterns
        patterns = [
            (r'from bsk_rl import act, data, obs, scene, sats', '# BSK-RL imports handled by setup helper'),
            (r'from bsk_rl\.sim import dyn, fsw', ''),
            (r'from Basilisk\.architecture import bskLogging', ''),
        ]
        
        for pattern, repl in patterns:
            new_content = re.sub(pattern, repl, new_content)
        
        # Add our import at the top after other imports
        if 'import gymnasium as gym' in new_content and 'from bsk_rl_setup import' not in new_content:
            new_content = new_content.replace(
                'import gymnasium as gym',
                'import gymnasium as gym\n\n# Import BSK-RL using setup helper\nfrom bsk_rl_setup import get_bsk_rl_imports\nact, data, obs, scene, sats, dyn, fsw, bskLogging = get_bsk_rl_imports()'
            )
    
    # Write back if changed
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"🔧 Fixed {file_path}")
    else:
        print(f"ℹ️  No changes needed for {file_path}")

def main():
    """Fix all Python files in the current directory"""
    files_to_fix = [
        'final_working_system.py',
        'proof_verification.py', 
        'test_suite.py',
        'working_bsk_collision_system.py',
        'working_demo.py'
    ]
    
    for file_name in files_to_fix:
        if os.path.exists(file_name):
            fix_bsk_rl_imports(file_name)
        else:
            print(f"⚠️  File not found: {file_name}")

if __name__ == "__main__":
    main()