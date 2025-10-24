#!/usr/bin/env python3
"""
Check Modern RLHF Framework
===========================

Simple check to verify the framework is working.
"""

import sys
import os
from pathlib import Path

print("🔍 Checking Modern RLHF Framework...")
print("=" * 50)

# Check if modern_rlhf directory exists
modern_rlhf_path = Path("modern_rlhf")
if not modern_rlhf_path.exists():
    print("❌ modern_rlhf directory not found!")
    sys.exit(1)

print("✅ modern_rlhf directory found")

# Check if all required files exist
required_files = [
    "__init__.py",
    "config.py", 
    "metrics.py",
    "reward_model.py",
    "trainer.py",
    "pipeline.py",
    "data_loader.py",
    "main.py",
    "requirements.txt",
    "README.md"
]

missing_files = []
for file in required_files:
    file_path = modern_rlhf_path / file
    if not file_path.exists():
        missing_files.append(file)

if missing_files:
    print(f"❌ Missing files: {missing_files}")
    sys.exit(1)

print("✅ All required files found")

# Try to import basic modules
try:
    sys.path.insert(0, str(modern_rlhf_path))
    
    print("🧪 Testing imports...")
    
    # Test config
    from config import ModernRLHFConfig, get_research_config
    print("✅ Config imports successful")
    
    # Test data loader
    from data_loader import ModernDataLoader
    print("✅ Data loader imports successful")
    
    # Test metrics
    from metrics import ModernMetricsEvaluator
    print("✅ Metrics imports successful")
    
    # Test configuration creation
    config = get_research_config()
    print("✅ Configuration creation successful")
    
    # Test data loader creation
    data_loader = ModernDataLoader(config)
    print("✅ Data loader creation successful")
    
    # Test synthetic data generation
    synthetic_data = data_loader._generate_synthetic_data()
    print(f"✅ Generated {len(synthetic_data)} synthetic samples")
    
    print("\n🎉 All checks passed! The Modern RLHF framework is ready to use.")
    print("\n📝 Next steps:")
    print("1. Install dependencies: pip install -r modern_rlhf/requirements.txt")
    print("2. Run quick test: python run_modern_rlhf.py")
    print("3. Run full training: python modern_rlhf/main.py --mode fast")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
