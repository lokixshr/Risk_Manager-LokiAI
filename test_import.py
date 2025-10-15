#!/usr/bin/env python3
"""
Simple test script to check if the Risk Manager application can be imported
"""

try:
    print("Testing application imports...")
    
    # Test basic imports
    import app.config
    print("✅ Config module imported successfully")
    
    import app.models
    print("✅ Models module imported successfully")
    
    import app.database
    print("✅ Database module imported successfully")
    
    # Test main app import
    import app.main
    print("✅ Main application module imported successfully")
    
    print("\n🎉 All critical modules imported successfully!")
    print("The application structure appears to be correct.")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()