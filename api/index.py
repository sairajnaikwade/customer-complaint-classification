import os
import sys

# Add project root to sys.path so app and src modules resolve properly in Vercel runtime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
