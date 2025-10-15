"""
Comprehensive load testing suite for Risk Manager Agent
Uses Locust for distributed load testing with realistic DeFi scenarios
"""
import random
import time
import json
import argparse
from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer
from locust.log import setup_logging
import gevent

# Sample wallet addresses for testing
SAMPLE_WALLETS = [
    "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5",
    "0x8ba1f109551bD432803012645Hac136c22C501e",
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",
    "0x1234567890123456789012345678901234567890",
    "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
    "0x5555555555555555555555555555555555555555",
    "0x9999999999999999999999999999999999999999",
    "0x1111111111111111111111111111111111111111",
    "0x2222222222222222222222222222222222222222",
    "0x3333333333333333333333333333333333333333"
]

class RiskManagerUser(HttpUser):
    """Simulates a user interacting with the Risk Manager API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.wallet_address = random.choice(SAMPLE_WALLETS)
        self.headers = {
            "x-wallet-address": self.wallet_address,
            "Content-Type": "application/json",
            "User-Agent": "LoadTest/1.0"
        }
    
    @task(10)
    def get_risk_summary(self):
        """Most common operation - get risk summary"""
        with self.client.get(
            f"/api/risk/summary?wallet={self.wallet_address}",
            headers=self.headers,
            catch_response=True,
            name="risk_summary"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "wallet_address" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(5)
    def get_alerts(self):
        """Get alerts for wallet"""
        params = {
            "limit": random.randint(10, 50),
            "skip": random.randint(0, 100)
        }
        
        # Sometimes add filters
        if random.random() < 0.3:
            params["severity"] = random.choice(["low", "medium", "high", "critical"])
        
        if random.random() < 0.2:
            params["resolved"] = random.choice([True, False])
        
        with self.client.get(
            "/api/risk/alerts",
            params=params,
            headers=self.headers,
            catch_response=True,
            name="get_alerts"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "total_count" in data and "alerts" in data:
                        response.success()
                    else:
                        response.failure("Invalid alerts response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def analyze_wallet(self):
        """Trigger wallet analysis"""
        payload = {
            "wallet_address": self.wallet_address,
            "force_refresh": random.choice([True, False]),
            "protocols": random.sample(["aave_v3", "compound", "curve"], k=random.randint(1, 2))
        }
        
        with self.client.post(
            "/api/risk/analyze",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="analyze_wallet"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and data["status"] == "completed":
                        response.success()
                    else:
                        response.failure("Analysis not completed")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def resolve_alert(self):
        """Simulate resolving an alert"""
        # Generate a fake ObjectId-like string
        fake_alert_id = "507f1f77bcf86cd799439011"
        
        with self.client.post(
            f"/api/risk/alerts/{fake_alert_id}/resolve",
            headers=self.headers,
            catch_response=True,
            name="resolve_alert"
        ) as response:
            # We expect 404 for fake ID, which is normal
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_system_status(self):
        """Check system status"""
        with self.client.get(
            "/api/risk/status",
            catch_response=True,
            name="system_status"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and "tracked_wallets" in data:
                        response.success()
                    else:
                        response.failure("Invalid status response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get(
            "/api/risk/health",
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code in [200, 503]:  # Both healthy and unhealthy are valid responses
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class SpikeTestUser(HttpUser):
    """User for spike testing - more aggressive load"""
    
    wait_time = between(0.1, 0.5)  # Much shorter wait time
    
    def on_start(self):
        self.wallet_address = random.choice(SAMPLE_WALLETS)
        self.headers = {
            "x-wallet-address": self.wallet_address,
            "Content-Type": "application/json"
        }
    
    @task
    def rapid_fire_requests(self):
        """Rapid fire requests to test rate limiting"""
        endpoints = [
            f"/api/risk/summary?wallet={self.wallet_address}",
            "/api/risk/alerts",
            "/api/risk/health"
        ]
        
        endpoint = random.choice(endpoints)
        
        with self.client.get(
            endpoint,
            headers=self.headers,
            catch_response=True,
            name="spike_test"
        ) as response:
            if response.status_code in [200, 429]:  # Success or rate limited
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


def run_load_test(host, duration=300, users=50, spawn_rate=2, test_type="normal"):
    """Run load test programmatically"""
    
    setup_logging("INFO", None)
    
    # Choose user class based on test type
    user_class = SpikeTestUser if test_type == "spike" else RiskManagerUser
    
    # Create environment
    env = Environment(user_classes=[user_class])
    
    # Start web UI on port 8089 if available
    try:
        env.create_web_ui("127.0.0.1", 8089)
        print(f"Load test web UI available at http://127.0.0.1:8089")
    except OSError:
        print("Port 8089 not available, running without web UI")
    
    # Start stats printer
    gevent.spawn(stats_printer(env.stats))
    
    # Start load test
    env.runner.start(users, spawn_rate=spawn_rate)
    
    # Run for specified duration
    print(f"Running {test_type} load test for {duration} seconds with {users} users...")
    print(f"Target host: {host}")
    
    # Update the host for all users
    for user_class in env.user_classes:
        user_class.host = host
    
    gevent.sleep(duration)
    
    # Stop the runner
    env.runner.stop()
    
    # Print final stats
    print("\n" + "="*50)
    print("LOAD TEST RESULTS")
    print("="*50)
    
    stats = env.stats
    
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Failure rate: {stats.total.fail_ratio:.2%}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")
    print(f"RPS: {stats.total.current_rps:.2f}")
    
    # Print percentiles
    print(f"50%ile response time: {stats.total.get_response_time_percentile(0.5):.2f}ms")
    print(f"95%ile response time: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99%ile response time: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    
    # Detailed stats by endpoint
    print("\nDETAILED STATS BY ENDPOINT:")
    print("-" * 50)
    for name, entry in stats.entries.items():
        print(f"{name:<20} | Requests: {entry.num_requests:>6} | "
              f"Failures: {entry.num_failures:>6} | "
              f"Avg: {entry.avg_response_time:>6.1f}ms | "
              f"95%: {entry.get_response_time_percentile(0.95):>6.1f}ms")
    
    # Check if test passed
    success = (
        stats.total.fail_ratio < 0.05 and  # Less than 5% failure rate
        stats.total.avg_response_time < 2000 and  # Average response time under 2s
        stats.total.get_response_time_percentile(0.95) < 5000  # 95%ile under 5s
    )
    
    print(f"\nLOAD TEST STATUS: {'PASSED' if success else 'FAILED'}")
    
    return success


def run_stress_test(host, duration=600):
    """Run stress test with gradually increasing load"""
    print("Starting stress test with gradually increasing load...")
    
    phases = [
        (50, 2, 120),    # 50 users for 2 minutes
        (100, 5, 120),   # 100 users for 2 minutes
        (200, 10, 120),  # 200 users for 2 minutes
        (400, 20, 120),  # 400 users for 2 minutes
        (800, 40, 120),  # 800 users for 2 minutes
    ]
    
    for users, spawn_rate, phase_duration in phases:
        print(f"\nPhase: {users} users, spawn rate: {spawn_rate}/s, duration: {phase_duration}s")
        success = run_load_test(host, phase_duration, users, spawn_rate)
        
        if not success:
            print(f"Stress test failed at {users} users")
            return False
        
        # Brief pause between phases
        time.sleep(10)
    
    print("\nStress test completed successfully!")
    return True


def run_endurance_test(host, duration=3600):
    """Run endurance test with steady load"""
    print(f"Starting endurance test for {duration} seconds (1 hour)...")
    return run_load_test(host, duration, users=100, spawn_rate=2)


def main():
    parser = argparse.ArgumentParser(description="Risk Manager Load Testing Suite")
    parser.add_argument("--host", default="http://localhost:8001", help="Target host")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=2, help="User spawn rate per second")
    parser.add_argument("--test-type", choices=["normal", "spike", "stress", "endurance"], 
                       default="normal", help="Type of load test")
    
    args = parser.parse_args()
    
    print(f"Risk Manager Agent Load Testing")
    print(f"Target: {args.host}")
    print(f"Test Type: {args.test_type}")
    
    success = False
    
    try:
        if args.test_type == "normal":
            success = run_load_test(args.host, args.duration, args.users, args.spawn_rate)
        elif args.test_type == "spike":
            success = run_load_test(args.host, args.duration, args.users, args.spawn_rate, "spike")
        elif args.test_type == "stress":
            success = run_stress_test(args.host)
        elif args.test_type == "endurance":
            success = run_endurance_test(args.host, args.duration)
        
        exit_code = 0 if success else 1
        print(f"\nExiting with code {exit_code}")
        exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nLoad test failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main()