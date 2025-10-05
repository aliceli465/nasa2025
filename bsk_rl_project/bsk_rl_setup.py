#!/usr/bin/env python3
"""
BSK-RL Setup Helper
Handles importing BSK-RL from different environments
"""
import sys
import os

def setup_bsk_rl():
    """Setup BSK-RL imports for the project"""
    
    # Try standard import first (if BSK-RL is installed)
    try:
        from bsk_rl import act, data, obs, scene, sats
        from bsk_rl.sim import dyn, fsw
        from Basilisk.architecture import bskLogging
        print("✅ Using installed BSK-RL package")
        return True
    except ImportError:
        pass
    
    # Try to find BSK-RL in the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Check if we're in the main nasa2025 repo with bsk_rl subdirectory
    bsk_rl_src = os.path.join(parent_dir, 'bsk_rl', 'src')
    if os.path.exists(bsk_rl_src):
        sys.path.insert(0, bsk_rl_src)
        try:
            from bsk_rl import act, data, obs, scene, sats
            from bsk_rl.sim import dyn, fsw
            from Basilisk.architecture import bskLogging
            print(f"✅ Using BSK-RL from: {bsk_rl_src}")
            return True
        except ImportError:
            pass
    
    # Check if BSK-RL is installed via pip in the bsk_rl directory
    bsk_rl_dir = os.path.join(parent_dir, 'bsk_rl')
    if os.path.exists(bsk_rl_dir):
        # Add the bsk_rl directory to sys.path and try to import
        sys.path.insert(0, bsk_rl_dir)
        try:
            from src.bsk_rl import act, data, obs, scene, sats
            from src.bsk_rl.sim import dyn, fsw
            from Basilisk.architecture import bskLogging
            print(f"✅ Using BSK-RL from: {bsk_rl_dir}")
            return True
        except ImportError:
            pass
    
    # Installation instructions
    print("❌ BSK-RL not found!")
    print("Please install BSK-RL using one of these methods:")
    print("")
    print("Option 1 - Install from the cloned bsk_rl directory:")
    print("  cd ../bsk_rl")
    print("  pip install -e .")
    print("")
    print("Option 2 - Run scripts from the bsk_rl directory:")
    print("  cd ../bsk_rl")
    print("  python ../bsk_rl_project/complete_real_bsk_system.py")
    print("")
    return False

def get_bsk_rl_imports():
    """Get BSK-RL imports after setup"""
    if setup_bsk_rl():
        from bsk_rl import act, data, obs, scene, sats
        from bsk_rl.sim import dyn, fsw
        from Basilisk.architecture import bskLogging
        return act, data, obs, scene, sats, dyn, fsw, bskLogging
    else:
        sys.exit(1)

if __name__ == "__main__":
    setup_bsk_rl()