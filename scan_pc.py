#!/usr/bin/env python3
"""
Hardware Test Suite - Comprehensive PC Hardware Testing Tool
Author: Hardware Tinkerer (Improved by Gemini)
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

# Core dependencies check
try:
    import psutil
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.prompt import Prompt
except ImportError:
    print("\033[91m[ERROR]\033[0m Core dependencies (psutil, rich) are not installed.")
    print("Please run: pip install psutil rich")
    sys.exit(1)

# Initialize Rich Console
console = Console()

class SystemChecker:
    """Handle system dependency checks and installations."""
    
    REQUIRED_PACKAGES = [
        'stress-ng', 'smartmontools', 'lm-sensors', 'hdparm',
        'fio', 'glmark2', 'speedtest-cli', 'x11-utils'  # x11-utils for xrandr
    ]
    
    def __init__(self):
        self.missing_packages = []

    def _is_package_installed(self, package):
        try:
            result = subprocess.run(['dpkg', '-s', package], capture_output=True, text=True, check=False)
            return 'Status: install ok installed' in result.stdout
        except FileNotFoundError:
            return False

    def check_dependencies(self):
        console.print("[cyan][INFO][/cyan] Checking system dependencies...")
        for package in self.REQUIRED_PACKAGES:
            if not self._is_package_installed(package):
                self.missing_packages.append(package)
        
        if self.missing_packages:
            console.print(f"[yellow][WARN][/yellow] Missing packages: {', '.join(self.missing_packages)}")
            return False
        console.print("[green][OK][/green] All dependencies satisfied.")
        return True

    def install_missing_packages(self):
        if not self.missing_packages:
            return True
        console.print("[cyan][INSTALL][/cyan] Installing missing packages...")
        try:
            subprocess.run(['apt', 'update'], check=True)
            for package in self.missing_packages:
                console.print(f"[blue][INFO][/blue] Installing {package}...")
                subprocess.run(['apt', 'install', '-y', package], check=True, capture_output=True, text=True)
                console.print(f"[green][OK][/green] {package} installed successfully.")
            self.missing_packages = []
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            console.print(f"[red][ERROR][/red] Failed to install packages: {e}")
            return False

class SystemInfo:
    """Gather and display system information."""
    
    def gather_info(self):
        console.print("[cyan][INFO][/cyan] Gathering system information...")
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._get_cpu_info(),
            'memory': self._get_memory_info(),
            'storage': self._get_storage_info(),
            'gpu': self._get_gpu_info(),
            'monitor': self._get_monitor_info(),
        }

    def _get_cpu_info(self):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        model = line.split(':')[1].strip()
                        break
            return {'model': model, 'cores': psutil.cpu_count(logical=True)}
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
            lsblk_out = subprocess.run(['lsblk', '-dno', 'NAME,MODEL'], capture_output=True, text=True).stdout
            models = {line.split()[0]: ' '.join(line.split()[1:]) for line in lsblk_out.strip().split('\n')}
            for p in psutil.disk_partitions(all=False):
                if p.device.startswith('/dev/sd') or p.device.startswith('/dev/nvme'):
                    disk_name = re.sub(r'p\d+$|\d+$', '', Path(p.device).name)
                    if disk_name not in [d['disk'] for d in devices]:
                        usage = psutil.disk_usage(p.mountpoint)
                        devices.append({
                            'disk': disk_name, 'device': f"/dev/{disk_name}",
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

    def _get_monitor_info(self):
        try:
            xrandr_out = subprocess.run(['xrandr'], capture_output=True, text=True).stdout
            res_search = re.search(r' connected primary (\d+x\d+)', xrandr_out) or re.search(r' connected (\d+x\d+)', xrandr_out)
            if not res_search:
                return "Unknown"
            
            resolution = res_search.group(1)
            
            # Find the line with the current mode for that resolution
            mode_line_search = re.search(fr'^\s*{resolution}\s+.*?(\d+\.\d+)\*\+?.*', xrandr_out, re.MULTILINE)
            
            if mode_line_search:
                refresh_rate = mode_line_search.group(1)
                return f"{resolution} @ {refresh_rate}Hz"
            else:
                # Fallback for cases where the mode line is not found as expected
                mode_line_search = re.search(fr'^\s*{resolution}\s+.*?(\d+\.\d+)\*.*', xrandr_out, re.MULTILINE)
                if mode_line_search:
                    refresh_rate = mode_line_search.group(1)
                    return f"{resolution} @ {refresh_rate}Hz"
                return resolution
        except Exception:
            return "Could not determine"

    def display_summary(self, info):
        table = Table(title="System Information Summary", border_style="green")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="magenta")
        
        table.add_row("CPU", f"{info['cpu']['model']} ({info['cpu']['cores']} Cores)")
        table.add_row("GPU", info['gpu'])
        table.add_row("Memory", f"{info['memory']['total_gb']} GB")
        storage_str = "\n".join([f"Disk {i}: {d['model']} ({d['size_gb']} GB)" for i, d in enumerate(info['storage'])])
        table.add_row("Storage", storage_str)
        table.add_row("Monitor", info['monitor'])
        
        console.print(Panel(table))

class TestRunner:
    """Main test orchestrator."""
    
    def __init__(self, config_path='config.json', args=None):
        self.args = args or argparse.Namespace()
        self.config = self._load_config(config_path)
        self.results = {'test_results': {}}
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.monitoring_data = {}
        self.stop_monitoring = threading.Event()

    def _load_config(self, config_path):
        default_config = {
            'test_duration': 60, 'storage_test_size': '1G',
            'temp_threshold_cpu': 90, 'temp_threshold_gpu': 85,
        }
        if self.args.quick:
            default_config['test_duration'] = 30
        
        try:
            with open(config_path, 'r') as f:
                default_config.update(json.load(f))
        except FileNotFoundError:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        return default_config

    def _monitor_worker(self):
        self.monitoring_data = {'max_cpu_temp': 0, 'max_gpu_temp': 0}
        while not self.stop_monitoring.is_set():
            # CPU Temp
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps and temps['coretemp']:
                    self.monitoring_data['max_cpu_temp'] = max(self.monitoring_data['max_cpu_temp'], max(t.current for t in temps['coretemp']))
            except Exception: pass

            # GPU Temp (universal approach)
            gpu_temp_found = False
            try: # 1. NVIDIA
                gpu_out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
                if gpu_out.strip():
                    self.monitoring_data['max_gpu_temp'] = max(self.monitoring_data['max_gpu_temp'], int(gpu_out.strip()))
                    gpu_temp_found = True
            except Exception: pass
            
            if not gpu_temp_found: # 2. AMD/Intel via psutil/lm-sensors
                try:
                    temps = psutil.sensors_temperatures()
                    for key, sensors in temps.items():
                        if 'amdgpu' in key or 'radeon' in key:
                            if sensors:
                                self.monitoring_data['max_gpu_temp'] = max(self.monitoring_data['max_gpu_temp'], sensors[0].current)
                                break
                except Exception: pass
            
            time.sleep(1)

    def _run_test_with_monitoring(self, cmd, duration, test_key):
        self.stop_monitoring.clear()
        monitor_thread = threading.Thread(target=self._monitor_worker)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        monitor_thread.start()
        
        try:
            # Wait for the specified duration
            time.sleep(duration)
            # Terminate the process gracefully
            process.terminate()
            # Wait for the process to terminate and get its output
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        finally:
            self.stop_monitoring.set()
            monitor_thread.join()

        temp_key = 'max_gpu_temp' if 'gpu' in test_key else 'max_cpu_temp'
        threshold = self.config['temp_threshold_gpu'] if 'gpu' in test_key else self.config['temp_threshold_cpu']
        
        # For glmark2, a return code of 0 or 143 (terminated) is acceptable
        passed = (process.returncode == 0 or (test_key == 'gpu_stress' and process.returncode == 143)) and self.monitoring_data[temp_key] <= threshold
        self.results['test_results'][test_key].update(self.monitoring_data)
        self.results['test_results'][test_key]['passed'] = passed
        
        status = f"[green]PASSED[/green]. Max Temp: {self.monitoring_data[temp_key]}°C" if passed else f"[red]FAILED[/red]. Max Temp: {self.monitoring_data[temp_key]}°C"
        console.print(f"[bold cyan][{test_key.split('_')[0].upper()}][/bold cyan] Test {status}")
        return passed

    def _run_simple_test(self, cmd, duration, test_key):
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=duration + 10)
            self.results['test_results'][test_key]['passed'] = True
            console.print(f"[bold cyan][{test_key.split('_')[0].upper()}][/bold cyan] Test [green]PASSED[/green].")
            return True
        except Exception as e:
            self.results['test_results'][test_key]['passed'] = False
            console.print(f"[bold cyan][{test_key.split('_')[0].upper()}][/bold cyan] Test [red]FAILED[/red]: {e}")
            return False

    def _test_cpu(self, duration):
        self.results['test_results']['cpu_stress'] = {}
        cmd = ['stress-ng', '--cpu', '0', '--timeout', f"{duration}s", '--metrics-brief']
        return self._run_test_with_monitoring(cmd, duration, 'cpu_stress')

    def _test_gpu(self, duration):
        self.results['test_results']['gpu_stress'] = {}
        cmd = ['glmark2', '--run-forever']
        return self._run_test_with_monitoring(cmd, duration, 'gpu_stress')

    def _test_memory(self, duration):
        self.results['test_results']['memory_stress'] = {}
        mem_size = '2G' if not self.args.quick else '1G'
        cmd = ['stress-ng', '--vm', '1', '--vm-bytes', mem_size, '--timeout', f"{duration}s"]
        return self._run_simple_test(cmd, duration, 'memory_stress')

    def _test_storage_health(self):
        key = 'storage_health'
        console.print(f"[bold cyan][STORAGE][/bold cyan] Checking storage SMART health...")
        self.results['test_results'][key] = {'details': [], 'passed': True}
        for device in self.results['system_info']['storage']:
            try:
                result = subprocess.run(['smartctl', '-H', device['device']], capture_output=True, text=True, check=True)
                if 'PASSED' not in result.stdout and 'OK' not in result.stdout:
                    self.results['test_results'][key]['passed'] = False
            except Exception:
                self.results['test_results'][key]['passed'] = False
        status = "[green]PASSED[/green]" if self.results['test_results'][key]['passed'] else "[red]FAILED[/red]"
        console.print(f"[bold cyan][STORAGE][/bold cyan] SMART health check {status}.")
        return self.results['test_results'][key]['passed']

    def _test_storage_performance(self, duration):
        key = 'storage_perf'
        console.print(f"[bold cyan][STORAGE][/bold cyan] Running storage performance benchmark...")
        self.results['test_results'][key] = {}
        test_file = self.results_dir / 'fio_test_file'
        cmd = [
            'fio', '--name=random-rw', '--rw=randrw', f'--size={self.config["storage_test_size"]}',
            '--direct=1', '--iodepth=64', f'--runtime={duration}', f'--filename={test_file}',
            '--ioengine=libaio', '--group_reporting', '--output-format=json'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            fio_results = json.loads(result.stdout)
            read_iops = int(fio_results['jobs'][0]['read']['iops'])
            write_iops = int(fio_results['jobs'][0]['write']['iops'])
            self.results['test_results'][key].update({'read_iops': read_iops, 'write_iops': write_iops, 'passed': True})
            console.print(f"[green][STORAGE][/green] Performance test PASSED. Read IOPS: {read_iops}, Write IOPS: {write_iops}")
            return True
        except Exception as e:
            self.results['test_results'][key]['passed'] = False
            console.print(f"[red][STORAGE][/red] Performance test FAILED: {e}")
            return False
        finally:
            if test_file.exists():
                test_file.unlink()

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = self.results_dir / f"test_results_{timestamp}.json"
        summary_filename = self.results_dir / f"summary_report_{timestamp}.txt"

        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        console.print(f"\n[blue][INFO][/blue] Full JSON report saved to: {json_filename}")

        summary = self._generate_summary_report()
        with open(summary_filename, 'w') as f:
            f.write(re.sub(r'\[(.*?)\]', '', summary)) # Remove rich tags for txt file
        console.print(f"[blue][INFO][/blue] Summary report saved to: {summary_filename}")

    def _generate_summary_report(self):
        info = self.results['system_info']
        res = self.results['test_results']
        
        s = f"===== Hardware Diagnostic Report =====\n"
        s += f"Timestamp: {info['timestamp']}\n\n"
        s += "-- System Information --\n"
        s += f"CPU: {info['cpu']['model']} ({info['cpu']['cores']} Cores)\n"
        s += f"GPU: {info['gpu']}\n"
        s += f"Memory: {info['memory']['total_gb']} GB\n"
        for i, disk in enumerate(info['storage']):
            s += f"Storage {i}: {disk['model']} ({disk['size_gb']} GB)\n"
        s += "\n-- Test Results --\n"

        overall_passed = all(data.get('passed', False) for data in res.values())
        for key, data in res.items():
            status = "[green][PASS][/green]" if data.get('passed') else "[red][FAIL][/red]"
            details = ""
            if 'cpu_stress' in key: details = f"(Max Temp: {data.get('max_cpu_temp', 'N/A')}°C)"
            elif 'gpu_stress' in key: details = f"(Max Temp: {data.get('max_gpu_temp', 'N/A')}°C)"
            elif 'storage_perf' in key: details = f"(Read: {data.get('read_iops', 'N/A')} IOPS, Write: {data.get('write_iops', 'N/A')} IOPS)"
            s += f"{status} {key.replace('_', ' ').title()} {details}\n"

        overall_status = "[bold green]ALL TESTS PASSED[/bold green]" if overall_passed else "[bold red]ONE OR MORE TESTS FAILED[/bold red]"
        s += f"\n-- Overall Status: {overall_status} --\n"
        
        console.print(Panel(s, title="Test Summary", border_style="blue"))
        return s

    def run_full_test(self):
        console.print(Panel("[bold cyan]Hardware Test Suite[/bold cyan]", expand=False))

        checker = SystemChecker()
        if not checker.check_dependencies():
            if Prompt.ask("Install missing packages?", choices=["y", "n"], default="y") == 'y':
                if not checker.install_missing_packages():
                    console.print("[red][ERROR][/red] Cannot proceed due to installation failure.")
                    return
            else:
                console.print("[yellow][WARN][/WARN] Continuing without all dependencies. Some tests may fail.")

        sys_info_comp = SystemInfo()
        self.results['system_info'] = sys_info_comp.gather_info()
        sys_info_comp.display_summary(self.results['system_info'])

        try:
            duration_str = Prompt.ask(f"Enter test duration in seconds", default=str(self.config['test_duration']))
            duration = int(duration_str)
        except (ValueError, TypeError):
            duration = self.config['test_duration']
            console.print(f"[yellow]Invalid input. Using default duration: {duration}s.[/yellow]")

        tests_to_run = {
            'cpu': (lambda: self._test_cpu(duration), not self.args.skip_cpu),
            'gpu': (lambda: self._test_gpu(duration), not self.args.skip_gpu),
            'memory': (lambda: self._test_memory(duration), not self.args.skip_cpu),
            'storage_health': (self._test_storage_health, not self.args.skip_storage),
            'storage_perf': (lambda: self._test_storage_performance(duration), not self.args.skip_storage),
        }

        with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            total_tests = sum(1 for _, should_run in tests_to_run.values() if should_run)
            task = progress.add_task("[green]Running tests...", total=total_tests)
            
            for test_name, (test_func, should_run) in tests_to_run.items():
                if should_run:
                    try:
                        progress.update(task, description=f"Running {test_name.replace('_', ' ')} test...")
                        test_func()
                    except Exception as e:
                        console.print(f"[red][ERROR][/red] Test '{test_name}' crashed: {e}")
                        self.results['test_results'][test_name] = {'passed': False, 'error': str(e)}
                    progress.advance(task)
                else:
                    console.print(f"[yellow][SKIP][/yellow] Skipping {test_name} test as requested.")

        self._save_results()

def main():    parser = argparse.ArgumentParser(description='Hardware Test Suite')    parser.add_argument('--quick', action='store_true', help='Run shorter tests (30s).')    parser.add_argument('--config', default='config.json', help='Path to config file.')    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU tests.')    parser.add_argument('--skip-storage', action='store_true', help='Skip all storage tests.')    parser.add_argument('--skip-cpu', action='store_true', help='Skip CPU and Memory tests.')    args = parser.parse_args()        try:        runner = TestRunner(config_path=args.config, args=args)        runner.run_full_test()    except KeyboardInterrupt:        console.print(f"\n[yellow][WARN][/yellow] Test interrupted by user.")    except Exception as e:        console.print(f"\n[red][ERROR][/red] An unexpected critical error occurred: {e}")        import traceback        traceback.print_exc()

if __name__ == "__main__":
    main()