
import time
import psutil
import GPUtil
import requests
import threading

from threading import Lock
from requests.exceptions import RequestException

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_types import ResourceProfile, ClusterResources
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Safety Planning")
printer = PrettyPrinter

class ResourceMonitor:
    """Real-time cluster resource tracking with failure resilience"""
    
    def __init__(self):
        self._node_cache = {}  # Cache last known good states
        self.config = load_global_config()

        self.resource_config = get_config_section('service_discovery')
        self.skip_localhost_http = self.resource_config.get('skip_localhost_http')
        self.static_nodes = self.resource_config.get('static_nodes')
        self.consul_url = self.resource_config.get('consul_url')
        self.k8s_token = self.resource_config.get('k8s_token')
        self.node_port = self.resource_config.get('node_port')
        self.k8s_api = self.resource_config.get('k8s_api')
        self.mode = self.resource_config.get('mode')

        self.cluster_resources = ClusterResources()
        self.update_interval = 5  # seconds
        self.node_query_timeout = 2  # seconds
        self._lock = Lock()
        self._init_monitoring_thread()
        self.allocations = {}
        self.resource_graph = {}

    def get_available_resources(self) -> ClusterResources:
        """Return current available cluster resources after subtracting allocations."""
        self._lock.acquire()
        try:
            allocated_gpu = 0
            allocated_ram = 0
            allocated_hw = set()
            
            for profile in self.cluster_resources.current_allocations.values():
                allocated_gpu += profile.gpu
                allocated_ram += profile.ram
                if profile.specialized_hardware:
                    allocated_hw.update(profile.specialized_hardware)
            
            available_gpu = self.cluster_resources.gpu_total - allocated_gpu
            available_ram = self.cluster_resources.ram_total - allocated_ram
            available_hw = [
                hw for hw in self.cluster_resources.specialized_hardware_available
                if hw not in allocated_hw
            ]
            
            return ClusterResources(
                gpu_total=available_gpu,
                ram_total=available_ram,
                specialized_hardware_available=available_hw,
                current_allocations={}
            )
        finally:
            self._lock.release()

    def _init_monitoring_thread(self):
        def monitor_loop():
            time.sleep(0.1)  # short delay to ensure full object init
            while True:
                self._update_resource_map()
                time.sleep(self.update_interval)

        threading.Thread(
            target=monitor_loop,
            daemon=True
        ).start()

    def _discover_cluster_nodes(self):
        """Discover nodes through configured service discovery backend"""
        try:
            if self.mode == 'consul':
                return self._query_consul_cluster()
            elif self.mode == 'k8s':
                return self._query_kubernetes_cluster()
            else:
                if not self.static_nodes:
                    return ['localhost']
                return self.static_nodes
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return ['localhost']

    def _query_consul_cluster(self):
        """Query Consul service discovery"""
        try:
            response = requests.get(
                f"{self.consul_url}/v1/catalog/nodes",
                timeout=self.node_query_timeout
            )
            response.raise_for_status()
            return [node['Node'] for node in response.json()]
        except RequestException as e:
            logger.error(f"Consul query failed: {str(e)}")
            return []

    def _query_kubernetes_cluster(self):
        """Query Kubernetes API for nodes"""
        try:
            headers = {"Authorization": f"Bearer {self.k8s_token}"}
            response = requests.get(
                f"{self.k8s_api}/api/v1/nodes",
                headers=headers,
                timeout=self.node_query_timeout
            )
            response.raise_for_status()
            return [item['metadata']['name'] for item in response.json()['items']]
        except RequestException as e:
            logger.error(f"Kubernetes API error: {str(e)}")
            return []

    def _query_node_resources(self, node_id):
        """Make RPC call to node's resource endpoint"""
        try:
            # First check cache for recent data
            if node_id in self._node_cache:
                cached = self._node_cache[node_id]
                if time.time() - cached['timestamp'] < self.update_interval * 2:
                    return cached['data']

            # Handle localhost directly without HTTP
            if node_id == "localhost" and self.skip_localhost_http:
                return self._get_local_resources()

            # Make HTTP call for other nodes
            response = requests.get(
                f"http://{node_id}:{self.node_port}/metrics",
                timeout=self.node_query_timeout
            )
            response.raise_for_status()
            
            # Parse response
            metrics = response.json()
            resource_data = {
                'gpu_available': metrics['gpu']['free'],
                'ram_available': metrics['memory']['free'],
                'specialized_hw': metrics.get('specialized_hw', []),
                'gpu_allocated': metrics['gpu']['total'] - metrics['gpu']['free'],
                'ram_allocated': metrics['memory']['total'] - metrics['memory']['free'],
                'specialized_allocated': metrics.get('specialized_allocated', [])
            }
            
            # Update cache
            self._node_cache[node_id] = {
                'data': resource_data,
                'timestamp': time.time()
            }
            
            return resource_data
            
        except RequestException as e:
            if node_id == "localhost":
                # Always fallback to local monitoring for localhost
                logger.debug(f"Using local resource fallback for {node_id}")
                return self._get_local_resources()
            else:
                logger.warning(f"Resource query failed for {node_id}: {str(e)}")
                if node_id in self._node_cache:
                    return self._node_cache[node_id]['data']
                return None

    def _get_local_resources(self):
        """Comprehensive local resource monitoring with GPU support"""
        try:
            mem = psutil.virtual_memory()
            gpu_count = 0

            # Try to detect NVIDIA GPUs
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
            except ImportError:
                pass

            return {
                'gpu_available': gpu_count,
                'ram_available': mem.available // (1024 ** 3),
                'specialized_hw': [],
                'gpu_allocated': 0,
                'ram_allocated': mem.used // (1024 ** 3),
                'specialized_allocated': []
            }
        except Exception as e:
            logger.error(f"Local resource check failed: {str(e)}")
            # Return safe defaults
            return {
                'gpu_available': 1,
                'ram_available': 16,
                'specialized_hw': [],
                'gpu_allocated': 0,
                'ram_allocated': 0,
                'specialized_allocated': []
            }

    def _update_resource_map(self):
        """Thread-safe resource map update"""
        self._lock.acquire()
        try:
            new_resources = ClusterResources(
                gpu_total=0,
                ram_total=0,
                specialized_hardware_available=[],
                current_allocations={}
            )
            
            nodes = self._discover_cluster_nodes()
            seen_hardware = set()
    
            if not nodes:
                logger.warning("No cluster nodes discovered")
                standalone_resources = ClusterResources(
                    gpu_total=1,
                    ram_total=32,
                    specialized_hardware_available=[],
                    current_allocations={}
                )
                if standalone_resources != self.cluster_resources:
                    self.cluster_resources = standalone_resources
                    # Show actual RAM value in log
                    logger.info(f"Using standalone resource profile: 1 GPU, {standalone_resources.ram_total} GB RAM")
                return
            
            for node_id in nodes:
                node_data = self._query_node_resources(node_id)
                if not node_data:
                    continue
                
                # Use get() with defaults for all keys
                new_resources.gpu_total += node_data.get('gpu_available', 0)
                new_resources.ram_total += node_data.get('ram_available', 0)
                
                # Handle specialized hardware safely
                for hw in node_data.get('specialized_hw', []):
                    if hw not in seen_hardware:
                        new_resources.specialized_hardware_available.append(hw)
                        seen_hardware.add(hw)
                
                # Store node allocation with safe access
                new_resources.current_allocations[node_id] = ResourceProfile(
                    gpu=node_data.get('gpu_allocated', 0),
                    ram=node_data.get('ram_allocated', 0),
                    specialized_hardware=node_data.get('specialized_allocated', [])
                )
            
            # Only update if significant changes occur
            if new_resources != self.cluster_resources:
                self.cluster_resources = new_resources
                logger.info("Cluster resource map updated")
        finally:
            self._lock.release()

    def allocate_resources(self, requirements: ResourceProfile):
        """Track resource consumption"""
        with self._lock:
            self.cluster_resources.gpu_total -= requirements.gpu
            self.cluster_resources.ram_total -= requirements.ram
            self.cluster_resources.specialized_hardware_available = [
                hw for hw in self.cluster_resources.specialized_hardware_available 
                if hw not in requirements.specialized_hardware
            ]
