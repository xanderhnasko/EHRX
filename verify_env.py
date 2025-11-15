"""
Verify environment variables are loaded correctly.
"""

import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=" * 80)
print("ENVIRONMENT VERIFICATION")
print("=" * 80)

# Check required variables
required_vars = {
    'GCP_PROJECT_ID': os.getenv('GCP_PROJECT_ID'),
    'GOOGLE_APPLICATION_CREDENTIALS': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
    'VLM_MODEL_NAME': os.getenv('VLM_MODEL_NAME'),
}

print("\n✓ Environment Variables Loaded:\n")
for var_name, var_value in required_vars.items():
    status = "✓" if var_value else "✗"
    print(f"  {status} {var_name}: {var_value}")

# Check credentials file exists
if required_vars['GOOGLE_APPLICATION_CREDENTIALS']:
    creds_path = Path(required_vars['GOOGLE_APPLICATION_CREDENTIALS'])
    if creds_path.exists():
        print(f"\n✓ Credentials file exists: {creds_path}")
        print(f"  Size: {creds_path.stat().st_size} bytes")
    else:
        print(f"\n✗ Credentials file NOT FOUND: {creds_path}")

# Try to initialize VLMConfig
print("\n" + "-" * 80)
print("Testing VLMConfig initialization...")
print("-" * 80 + "\n")

try:
    from ehrx.vlm.config import VLMConfig
    config = VLMConfig.from_env()
    print(f"✓ VLMConfig initialized successfully!")
    print(f"  {config}")
except Exception as e:
    print(f"✗ VLMConfig failed: {e}")

print("\n" + "=" * 80)
print("Environment verification complete!")
print("=" * 80)
