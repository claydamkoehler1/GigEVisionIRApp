#!/usr/bin/env python3
"""
Test script for Allen-Bradley PLC communication using pycomm3
This script tests reading the heating profile tags from the PLC
"""

import sys
import time
from datetime import datetime

# Try to import pycomm3
try:
    from pycomm3 import LogixDriver
    print("✅ pycomm3 imported successfully")
except ImportError:
    print("❌ Error: pycomm3 not installed")
    print("Install with: pip install pycomm3")
    sys.exit(1)

def test_plc_connection(ip_address="192.168.1.100"):
    """Test PLC connection and read heating profile tags"""
    
    print(f"\n🔌 Testing PLC connection to {ip_address}...")
    print("=" * 60)
    
    plc = None
    try:
        # Initialize PLC connection
        plc = LogixDriver(ip_address)
        print(f"📡 Opening connection to {ip_address}...")
        
        # Open connection
        plc.open()
        print("✅ Connection established successfully!")
        
        # Test basic connection with a simple read
        print("\n🔍 Testing basic connectivity...")
        test_result = plc.read("Heat_Seq_Time_SP[1]")
        
        if test_result.error is None:
            print(f"✅ Test read successful: Heat_Seq_Time_SP[1] = {test_result.value}")
        else:
            print(f"⚠️  Test read failed: {test_result.error}")
            print("This might be normal if the tag doesn't exist")
        
        print("\n📊 Reading complete heating profile...")
        print("-" * 60)
        
        # Read all heating profile tags
        valid_steps = 0
        for step in range(1, 16):  # Steps 1-15
            print(f"\n📋 Step {step}:")
            step_valid = True
            
            # Read time setpoint
            time_tag = f"Heat_Seq_Time_SP[{step}]"
            time_result = plc.read(time_tag)
            if time_result.error is None:
                print(f"  ⏱️  Time SP: {time_result.value} minutes")
            else:
                print(f"  ❌ Time SP: ERROR - {time_result.error}")
                step_valid = False
            
            # Read start temperature setpoint
            start_temp_tag = f"Heat_Seq_Start_Temp_SP[{step}]"
            start_temp_result = plc.read(start_temp_tag)
            if start_temp_result.error is None:
                print(f"  🌡️  Start Temp: {start_temp_result.value}°F")
            else:
                print(f"  ❌ Start Temp: ERROR - {start_temp_result.error}")
                step_valid = False
            
            # Read end temperature setpoint
            end_temp_tag = f"Heat_Seq_End_Temp_SP[{step}]"
            end_temp_result = plc.read(end_temp_tag)
            if end_temp_result.error is None:
                print(f"  🌡️  End Temp: {end_temp_result.value}°F")
            else:
                print(f"  ❌ End Temp: ERROR - {end_temp_result.error}")
                step_valid = False
            
            # Read vacuum setpoint
            vac_tag = f"Heat_Seq_Vac_SP[{step}]"
            vac_result = plc.read(vac_tag)
            if vac_result.error is None:
                print(f"  🌪️  Vacuum: {vac_result.value}")
            else:
                print(f"  ❌ Vacuum: ERROR - {vac_result.error}")
                step_valid = False
            
            if step_valid:
                valid_steps += 1
                print(f"  ✅ Step {step} - All tags read successfully")
            else:
                print(f"  ⚠️  Step {step} - Some tags failed")
        
        print("\n" + "=" * 60)
        print(f"📈 Summary: {valid_steps}/15 steps read successfully")
        
        if valid_steps > 0:
            print("✅ PLC communication is working!")
        else:
            print("⚠️  No steps were read successfully - check tag names and PLC configuration")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPossible issues:")
        print("- PLC IP address is incorrect")
        print("- PLC is not reachable (network/firewall)")
        print("- PLC is not running or configured for Ethernet/IP")
        print("- Tag names don't match PLC program")
        
    finally:
        # Close connection
        if plc:
            try:
                plc.close()
                print("\n🔒 PLC connection closed")
            except:
                pass

def test_bulk_read(ip_address="192.168.1.100"):
    """Test reading multiple tags at once for better performance"""
    
    print(f"\n🚀 Testing bulk read from {ip_address}...")
    print("=" * 60)
    
    plc = None
    try:
        plc = LogixDriver(ip_address)
        plc.open()
        
        # Create list of all tags to read
        tags_to_read = []
        for step in range(1, 16):
            tags_to_read.extend([
                f"Heat_Seq_Time_SP[{step}]",
                f"Heat_Seq_Start_Temp_SP[{step}]", 
                f"Heat_Seq_End_Temp_SP[{step}]",
                f"Heat_Seq_Vac_SP[{step}]"
            ])
        
        print(f"📡 Reading {len(tags_to_read)} tags in bulk...")
        start_time = time.time()
        
        # Read all tags at once
        results = plc.read(tags_to_read)
        
        read_time = time.time() - start_time
        print(f"⏱️  Bulk read completed in {read_time:.2f} seconds")
        
        # Process results
        successful_reads = 0
        for i, result in enumerate(results):
            tag_name = tags_to_read[i]
            if result.error is None:
                successful_reads += 1
                print(f"✅ {tag_name}: {result.value}")
            else:
                print(f"❌ {tag_name}: ERROR - {result.error}")
        
        print(f"\n📊 Bulk read summary: {successful_reads}/{len(tags_to_read)} tags successful")
        
    except Exception as e:
        print(f"❌ Bulk read failed: {e}")
        
    finally:
        if plc:
            try:
                plc.close()
            except:
                pass

if __name__ == "__main__":
    print("🏭 Allen-Bradley PLC Test Script")
    print("Using pycomm3 library")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get IP address from command line or use default
    ip_address = "192.168.1.100"  # Default
    if len(sys.argv) > 1:
        ip_address = sys.argv[1]
    
    print(f"🎯 Target PLC IP: {ip_address}")
    
    # Run individual tag test
    test_plc_connection(ip_address)
    
    # Wait a moment
    time.sleep(2)
    
    # Run bulk read test
    test_bulk_read(ip_address)
    
    print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Usage: python test_plc_connection.py [IP_ADDRESS]")
    print("   Example: python test_plc_connection.py 192.168.1.50")
