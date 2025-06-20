#!/usr/bin/env python3
"""
Hardware Test Suite - Comprehensive PC Hardware Testing Tool
Author: Hardware Tinkerer
Description: Automated testing suite for CPU, GPU, Storage, and Memory for new PCs.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import re
from datetime import datetime
from pathlib import Path

# Try to import psutil, which is a core dependency
try:
    import psutil
except ImportError:
    print("\033[91m[ERROR]\033[0m psutil is not installed. Please run: pip install psutil")
    sys.exit(1)

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class SystemChecker:
    """Handle system dependency checks and installations."""
    
    REQUIRED_PACKAGES = [
        'stress-ng',
        'smartmontools',
        'lm-sensors',
        'hdparm',
        'fio',
        'glmark2',
        'speedtest-cli',
        'python3-psutil'
    ]
    
    def __init__(self):
        self.missing_packages = []

    def _is_package_installed(self, package):
        """Check if a package is installed (for Debian-based systems)."""
        try:
            result = subprocess.run(['dpkg', '-s', package], capture_output=True, text=True, check=False)
            return 'Status: install ok installed' in result.stdout
        except FileNotFoundError:
            return False

    def _is_command_available(self, command):
        """Check if a command is available in PATH."""
        return subprocess.run(['which', command], capture_output=True).returncode == 0
    
    def check_dependencies(self):
        """Check if required packages are installed."""
        print(f"{Colors.BLUE}[INFO]{Colors.END} Checking system dependencies...")
        for package in self.REQUIRED_PACKAGES:
            if not self._is_package_installed(package):
                self.missing_packages.append(package)
        
        if self.missing_packages:
            print(f"{Colors.YELLOW}[WARN]{Colors.END} Missing packages: {', '.join(self.missing_packages)}")
            return False
        print(f"{Colors.GREEN}[OK]{Colors.END} All dependencies satisfied.")
        return True

    def install_missing_packages(self):
        """Install missing packages using apt."""
        if not self.missing_packages:
            return True
        print(f"{Colors.CYAN}[INSTALL]{Colors.END} Installing missing packages...")
        try:
            subprocess.run(['apt', 'update'], check=True)
            for package in self.missing_packages:
                print(f"{Colors.BLUE}[INFO]{Colors.END} Installing {package}...")
                install_cmd = ['apt', 'install', '-y', package]
                if package == 'speedtest-cli':
                    # The official speedtest CLI requires accepting a license
                    print(f"{Colors.YELLOW}[WARN]{Colors.END} The 'speedtest' command will ask for license agreement on first run.")
                subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                print(f"{Colors.GREEN}[OK]{Colors.END} {package} installed successfully.")
            self.missing_packages = []
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"{Colors.RED}[ERROR]{Colors.END} Failed to install packages: {e}")
            return False

class SystemInfo:
    """Gather system information."""
    
    def gather_info(self):
        """Collect and return a dictionary of system information."""
        print(f"{Colors.BLUE}[INFO]{Colors.END} Gathering system information...")
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._get_cpu_info(),
            'memory': self._get_memory_info(),
            'storage': self._get_storage_info(),
            'gpu': self._get_gpu_info(),
        }

    def _get_cpu_info(self):
        model = "Unknown"
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        model = line.split(':')[1].strip()
                        break
            cores = psutil.cpu_count(logical=True)
            return {'model': model, 'cores': cores}
        except Exception:
            return {'model': 'Unknown', 'cores': 0}

    def _get_memory_info(self):
        try:
            mem = psutil.virtual_memory()
            return {'total_gb': round(mem.total / (1024**3), 1)}
        except Exception:
            return {'total_gb': 0}

    def _get_storage_info(self):
        devices = []
        try:
            partitions = psutil.disk_partitions(all=False)
            # Use lsblk to get model info, as psutil does not provide it
            lsblk_out = subprocess.run(
                ['lsblk', '-dno', 'NAME,MODEL'],
                capture_output=True, text=True
            ).stdout
            models = {line.split()[0]: ' '.join(line.split()[1:]) for line in lsblk_out.strip().split('\n')}
            
            for p in partitions:
                if p.device.startswith('/dev/sd') or p.device.startswith('/dev/nvme'):
                    disk_name = re.sub(r'p\d+$|\d+$', '', Path(p.device).name)
                    if disk_name not in [d['disk'] for d in devices]:
                         usage = psutil.disk_usage(p.mountpoint)
                         devices.append({
                             'disk': disk_name,
                             'device': f"/dev/{disk_name}",
                             'size_gb': round(usage.total / (1024**3)),
                             'model': models.get(disk_name, 'Unknown')
                         })
            return devices
        except Exception:
            return []

    def _get_gpu_info(self):
        try:
            lspci_out = subprocess.run(['lspci'], capture_output=True, text=True, check=True).stdout
            for line in lspci_out.split('\n'):
                if 'VGA compatible controller' in line or '3D controller' in line:
                    return line.split(': ')[1].strip()
            return 'No dedicated GPU detected'
        except Exception:
            return 'Unknown'


class TestRunner:
    """Main test orchestrator."""
    
    def __init__(self, config_path='config.json', args=None):
        self.args = args or argparse.Namespace()
        self.config = self._load_config(config_path)
        self.results = { 'test_results': {} }
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.monitoring_data = {}
        self.stop_monitoring = threading.Event()

    def _load_config(self, config_path):
        """Load configuration from JSON file, updating defaults."""
        default_config = {
            'test_duration': 60,
            'storage_test_size': '1G',
            'temp_threshold_cpu': 90,
            'temp_threshold_gpu': 85,
            'quick_mode': False
        }
        if self.args.quick:
            default_config['quick_mode'] = True
            default_config['test_duration'] = 30
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        return default_config

    def _monitor_worker(self):
        """Worker thread to monitor temps and fan speeds."""
        self.monitoring_data = {'max_cpu_temp': 0, 'max_gpu_temp': 0, 'max_fan_speed': 0}
        
        while not self.stop_monitoring.is_set():
            # CPU Temp
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = max(t.current for t in temps['coretemp'])
                    self.monitoring_data['max_cpu_temp'] = max(self.monitoring_data['max_cpu_temp'], cpu_temp)
            except Exception: pass

            # GPU Temp (nvidia-smi for NVIDIA)
            try:
                gpu_out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
                gpu_temp = int(gpu_out.strip())
                self.monitoring_data['max_gpu_temp'] = max(self.monitoring_data['max_gpu_temp'], gpu_temp)
            except Exception: pass

            # Fan Speed
            try:
                fans = psutil.sensors_fans()
                if fans:
                    fan_speed = max(f.current for f in list(fans.values())[0])
                    self.monitoring_data['max_fan_speed'] = max(self.monitoring_data['max_fan_speed'], fan_speed)
            except Exception: pass

            time.sleep(1)

    def _run_test_with_monitoring(self, cmd, duration, test_key):
        """Run a subprocess test while monitoring thermals."""
        self.stop_monitoring.clear()
        monitor_thread = threading.Thread(target=self._monitor_worker)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        monitor_thread.start()
        
        try:
            stdout, stderr = process.communicate(timeout=duration + 10)
            self.stop_monitoring.set()
            monitor_thread.join()
            
            self.results['test_results'][test_key].update(self.monitoring_data)
            
            if process.returncode != 0:
                print(f"{Colors.RED}[FAIL]{Colors.END} Test process returned non-zero exit code.")
                print(stderr)
                return False
                
            temp_key = 'max_gpu_temp' if 'gpu' in test_key else 'max_cpu_temp'
            threshold = self.config['temp_threshold_gpu'] if 'gpu' in test_key else self.config['temp_threshold_cpu']
            
            if self.monitoring_data[temp_key] > threshold:
                print(f"{Colors.RED}[FAIL]{Colors.END} Thermal threshold exceeded! ({self.monitoring_data[temp_key]}°C > {threshold}°C)")
                return False

            return True

        except subprocess.TimeoutExpired:
            process.kill()
            self.stop_monitoring.set()
            monitor_thread.join()
            print(f"{Colors.RED}[FAIL]{Colors.END} Test timed out.")
            return False

    def _test_cpu(self):
        key = 'cpu_stress'
        print(f"{Colors.CYAN}[CPU]{Colors.END} Starting CPU stress test...")
        self.results['test_results'][key] = {}
        cmd = ['stress-ng', '--cpu', '0', '--timeout', f"{self.config['test_duration']}s", '--metrics-brief']
        passed = self._run_test_with_monitoring(cmd, self.config['test_duration'], key)
        self.results['test_results'][key]['passed'] = passed
        print(f"{Colors.GREEN if passed else Colors.RED}[CPU]{Colors.END} Test {'PASSED' if passed else 'FAILED'}. Max Temp: {self.monitoring_data.get('max_cpu_temp', 'N/A')}°C")
        return passed

    def _test_gpu(self):
        key = 'gpu_stress'
        print(f"{Colors.CYAN}[GPU]{Colors.END} Starting GPU stress test...")
        self.results['test_results'][key] = {}
        cmd = ['glmark2']
        passed = self._run_test_with_monitoring(cmd, self.config['test_duration'], key)
        self.results['test_results'][key]['passed'] = passed
        print(f"{Colors.GREEN if passed else Colors.RED}[GPU]{Colors.END} Test {'PASSED' if passed else 'FAILED'}. Max Temp: {self.monitoring_data.get('max_gpu_temp', 'N/A')}°C")
        return passed

    def _test_memory(self):
        key = 'memory_stress'
        print(f"{Colors.CYAN}[MEMORY]{Colors.END} Starting memory stress test...")
        self.results['test_results'][key] = {}
        mem_size = '2G' if not self.config['quick_mode'] else '1G'
        cmd = ['stress-ng', '--vm', '1', '--vm-bytes', mem_size, '--timeout', f"{self.config['test_duration']}s"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=self.config['test_duration'] + 10)
            self.results['test_results'][key]['passed'] = True
            print(f"{Colors.GREEN}[MEMORY]{Colors.END} Test PASSED.")
            return True
        except Exception as e:
            self.results['test_results'][key]['passed'] = False
            print(f"{Colors.RED}[MEMORY]{Colors.END} Test FAILED: {e}")
            return False

    def _test_storage_health(self):
        key = 'storage_health'
        print(f"{Colors.CYAN}[STORAGE]{Colors.END} Checking storage SMART health...")
        self.results['test_results'][key] = {'details': []}
        all_passed = True
        for device in self.results['system_info']['storage']:
            dev_path = device['device']
            try:
                result = subprocess.run(['smartctl', '-H', dev_path], capture_output=True, text=True, check=True)
                if 'PASSED' in result.stdout or 'OK' in result.stdout:
                    status = 'PASSED'
                else:
                    status = 'FAILED'
                    all_passed = False
                self.results['test_results'][key]['details'].append({'device': dev_path, 'status': status})
            except Exception as e:
                status = 'UNKNOWN'
                self.results['test_results'][key]['details'].append({'device': dev_path, 'status': status, 'error': str(e)})
        self.results['test_results'][key]['passed'] = all_passed
        print(f"{Colors.GREEN if all_passed else Colors.RED}[STORAGE]{Colors.END} SMART health check {'PASSED' if all_passed else 'FAILED'}.")
        return all_passed

    def _test_storage_performance(self):
        key = 'storage_perf'
        print(f"{Colors.CYAN}[STORAGE]{Colors.END} Running storage performance benchmark...")
        self.results['test_results'][key] = {}
        test_file = self.results_dir / 'fio_test_file'
        cmd = [
            'fio', '--name=random-rw', '--rw=randrw', f'--size={self.config["storage_test_size"]}',
            '--direct=1', '--iodepth=64', f'--runtime={self.config["test_duration"]}',
            f'--filename={test_file}', '--ioengine=libaio', '--group_reporting', '--output-format=json'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            fio_results = json.loads(result.stdout)
            read_iops = int(fio_results['jobs'][0]['read']['iops'])
            write_iops = int(fio_results['jobs'][0]['write']['iops'])
            self.results['test_results'][key].update({'read_iops': read_iops, 'write_iops': write_iops, 'passed': True})
            print(f"{Colors.GREEN}[STORAGE]{Colors.END} Performance test PASSED. Read IOPS: {read_iops}, Write IOPS: {write_iops}")
            return True
        except Exception as e:
            self.results['test_results'][key]['passed'] = False
            print(f"{Colors.RED}[STORAGE]{Colors.END} Performance test FAILED: {e}")
            return False
        finally:
            if test_file.exists():
                test_file.unlink()

    def _test_network(self):
        key = 'network'
        print(f"{Colors.CYAN}[NETWORK]{Colors.END} Running network tests...")
        self.results['test_results'][key] = {}
        passed = True
        # Ping test
        try:
            ping_result = subprocess.run(['ping', '-c', '4', '8.8.8.8'], capture_output=True, text=True, check=True)
            avg_latency = re.search(r'rtt min/avg/max/mdev = .*?/(.*?)/', ping_result.stdout).group(1)
            self.results['test_results'][key]['ping_latency_ms'] = float(avg_latency)
        except Exception:
            self.results['test_results'][key]['ping_latency_ms'] = 'FAIL'
            passed = False
        
        # Speedtest
        try:
            # Add --accept-license to run non-interactively
            speed_result = subprocess.run(['speedtest-cli', '--simple', '--accept-license'], capture_output=True, text=True, check=True)
            download = re.search(r'Download: ([\d.]+) Mbit/s', speed_result.stdout).group(1)
            upload = re.search(r'Upload: ([\d.]+) Mbit/s', speed_result.stdout).group(1)
            self.results['test_results'][key].update({'download_mbps': float(download), 'upload_mbps': float(upload)})
        except Exception:
             self.results['test_results'][key].update({'download_mbps': 'FAIL', 'upload_mbps': 'FAIL'})
             passed = False
        
        self.results['test_results'][key]['passed'] = passed
        print(f"{Colors.GREEN if passed else Colors.RED}[NETWORK]{Colors.END} Network test {'PASSED' if passed else 'FAILED'}.")
        return passed

    def _save_results(self):
        """Save results to JSON and generate a human-readable summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = self.results_dir / f"test_results_{timestamp}.json"
        summary_filename = self.results_dir / f"summary_report_{timestamp}.txt"

        # Save JSON
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n{Colors.BLUE}[INFO]{Colors.END} Full JSON report saved to: {json_filename}")

        # Generate and save summary
        summary = self._generate_summary_report()
        with open(summary_filename, 'w') as f:
            f.write(summary)
        print(f"{Colors.BLUE}[INFO]{Colors.END} Summary report saved to: {summary_filename}")

    def _generate_summary_report(self):
        """Creates a human-readable string summary of the test results."""
        info = self.results['system_info']
        res = self.results['test_results']
        
        s = f"===== Hardware Diagnostic Report =====\n"
        s += f"Timestamp: {info['timestamp']}\n\n"
        s += "-- System Information --\n"
        s += f"CPU: {info['cpu']['model']} ({info['cpu']['cores']} Cores)\n"
        s += f"GPU: {info['gpu']}\n"
        s += f"Memory: {info['memory']['total_gb']} GB\n"
        for i, disk in enumerate(info['storage']):
            s += f"Storage {i}: /dev/{disk['disk']} ({disk['size_gb']} GB, {disk['model']})\n"
        s += "\n-- Test Results --\n"

        overall_passed = True
        for key, data in res.items():
            status = f"{Colors.GREEN}[PASS]{Colors.END}" if data.get('passed') else f"{Colors.RED}[FAIL]{Colors.END}"
            if not data.get('passed'): overall_passed = False
            
            details = ""
            if 'cpu_stress' in key: details = f"(Max Temp: {data.get('max_cpu_temp', 'N/A')}°C, Max Fan: {data.get('max_fan_speed', 'N/A')} RPM)"
            elif 'gpu_stress' in key: details = f"(Max Temp: {data.get('max_gpu_temp', 'N/A')}°C)"
            elif 'storage_perf' in key: details = f"(Read: {data.get('read_iops', 'N/A')} IOPS, Write: {data.get('write_iops', 'N/A')} IOPS)"
            elif 'network' in key: details = f"(Ping: {data.get('ping_latency_ms', 'N/A')}ms, Down: {data.get('download_mbps', 'N/A')} Mbps)"
            
            s += f"{status} {key.replace('_', ' ').title()} {details}\n"

        overall_status = f"{Colors.GREEN}PASS{Colors.END}" if overall_passed else f"{Colors.RED}FAIL{Colors.END}"
        s += f"\n-- Overall Status: {overall_status} --\n"
        
        # Also print summary to console, removing color codes for the file version
        print(f"\n{Colors.BOLD}{Colors.CYAN}=== Test Summary ==={Colors.END}")
        print(re.sub(r'\033\[\d+m', '', s))
        
        return re.sub(r'\033\[\d+m', '', s)

    def run_full_test(self):
        """Run the complete hardware test suite."""
        print(f"{Colors.BOLD}{Colors.CYAN}=== Hardware Test Suite Starting ==={Colors.END}")

        checker = SystemChecker()
        if not checker.check_dependencies():
            if input("Install missing packages? (y/n): ").lower() == 'y':
                if not checker.install_missing_packages():
                    print(f"{Colors.RED}[ERROR]{Colors.END} Cannot proceed due to installation failure.")
                    return
            else:
                print(f"{Colors.YELLOW}[WARN]{Colors.END} Continuing without all dependencies. Some tests may fail.")

        self.results['system_info'] = SystemInfo().gather_info()
        
        # Define test execution map
        tests_to_run = {
            'cpu': (self._test_cpu, not self.args.skip_cpu),
            'gpu': (self._test_gpu, not self.args.skip_gpu),
            'memory': (self._test_memory, not self.args.skip_cpu),
            'storage_health': (self._test_storage_health, not self.args.skip_storage),
            'storage_perf': (self._test_storage_performance, not self.args.skip_storage),
            'network': (self._test_network, not self.args.skip_network)
        }

        for test_name, (test_func, should_run) in tests_to_run.items():
            if should_run:
                try:
                    test_func()
                except Exception as e:
                    print(f"{Colors.RED}[ERROR]{Colors.END} Test '{test_name}' crashed: {e}")
                    self.results['test_results'][test_name] = {'passed': False, 'error': str(e)}
            else:
                print(f"{Colors.YELLOW}[SKIP]{Colors.END} Skipping {test_name} test as requested.")

        self._save_results()

def main():
    """Main entry point for the script."""
    if os.geteuid() != 0:
        print(f"{Colors.CYAN}[INFO]{Colors.END} Relaunching with sudo for hardware access...")
        os.execvp('sudo', ['sudo', sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description='Hardware Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run shorter tests.')
    parser.add_argument('--config', default='config.json', help='Path to config file.')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU tests.')
    parser.add_argument('--skip-storage', action='store_true', help='Skip all storage tests.')
    parser.add_argument('--skip-network', action='store_true', help='Skip network tests.')
    parser.add_argument('--skip-cpu', action='store_true', help='Skip CPU and Memory tests.')
    args = parser.parse_args()
    
    try:
        runner = TestRunner(config_path=args.config, args=args)
        runner.run_full_test()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[WARN]{Colors.END} Test interrupted by user.")
    except Exception as e:
        print(f"\n{Colors.RED}[ERROR]{Colors.END} An unexpected critical error occurred: {e}")

if __name__ == "__main__":
    main()