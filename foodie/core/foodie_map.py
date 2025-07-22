
import json
import math
import time
import requests

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from foodie.utils.config_loader import load_global_config, get_config_section
from foodie.utils.error_handler import GeolocationError, ServiceAreaError, ConfigurationError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Foodie Map")
printer = PrettyPrinter

class FoodieMap:
    """
    Handles delivery route planning with traffic and weather considerations for Aruba.
    Integrates with external mapping services and weather APIs.
    """
    def __init__(self):
        self.config = load_global_config()
        self.central_hub = self.config.get('central_hub')

        self.map_config = get_config_section('foodie_map')
        if not self.central_hub:
            logger.warning("Central hub not configured, using default")
            self.central_hub = {"latitude": 12.516, "longitude": -70.035}

        self.map_config = get_config_section('foodie_map')
        self.avoid_tolls = self.map_config.get('avoid_tolls', True)
        self.api_key = self.map_config.get('api_key')
        self.base_url = self.map_config.get('base_url')
        self.traffic_endpoint = self.map_config.get('traffic_endpoint')
        self.weather_endpoint = self.map_config.get('weather_endpoint')
        self.route_endpoint = self.map_config.get('route_endpoint')

        required_keys = ['api_key', 'base_url', 'traffic_endpoint', 
                         'weather_endpoint', 'route_endpoint']
        for key in required_keys:
            if key not in self.map_config:
                raise ConfigurationError('foodie_map', key)

        logger.info("Foodie Map module initialized")

    def _validate_location(self, loc: Dict) -> bool:
        return (isinstance(loc, dict) and 
                -90 <= loc.get('latitude', 100) <= 90 and 
                -180 <= loc.get('longitude', 200) <= 180)

    @staticmethod
    def _locations_equal(loc1, loc2, tolerance=0.000001):
        try:
            lat1 = loc1.get('latitude')
            lon1 = loc1.get('longitude')
            lat2 = loc2.get('latitude')
            lon2 = loc2.get('longitude')
            
            if None in (lat1, lon1, lat2, lon2):
                return False
            
            return abs(lat1 - lat2) <= tolerance and abs(lon1 - lon2) <= tolerance
        except Exception:
            return False

    @staticmethod
    def update_location(state, new_location):
        """Updates current location in state"""
        if new_location:
            state['current_location'] = new_location

    def get_live_route(self, origin, destination, waypoints=None):
        """Get route with real-time traffic and weather"""
        # First get base route
        route = self.get_route(origin, destination, waypoints)
        
        # Get live traffic and weather
        live_traffic = self.get_traffic_data(destination)
        live_weather = self.get_weather_data(destination)
        
        # Apply real-time adjustments
        route = self._adjust_for_weather(route, live_weather)
        route = self._adjust_for_traffic(route, live_traffic)
        
        return route

    def get_route(self, origin: Dict[str, float], destination: Dict[str, float],
        waypoints: Optional[List[Dict[str, float]]] = None) -> Dict:
        """
        Get optimized delivery route considering traffic and weather

        Args:
            origin: {'latitude': x, 'longitude': y}
            destination: {'latitude': x, 'longitude': y}
            waypoints: Intermediate stops
            avoid_tolls: Whether to avoid toll roads

        Returns:
            Route information dictionary
        """
        # Validate coordinates
        self._validate_coordinates(origin)
        self._validate_coordinates(destination)

        # Check if within Aruba service area
        self._check_service_area(destination)

        # Get current traffic and weather
        traffic_data = self.get_traffic_data(destination)
        weather_data = self.get_weather_data(destination)

        # Build route request
        params = {
            'origin': f"{origin['latitude']},{origin['longitude']}",
            'destination': f"{destination['latitude']},{destination['longitude']}",
            'waypoints': '|'.join([f"{p['latitude']},{p['longitude']}" for p in waypoints]) if waypoints else '',
            'avoid': 'tolls' if self.avoid_tolls else '',
            'traffic_model': 'best_guess',
            'departure_time': 'now',
            'key': self.api_key
        }

        # Call mapping API
        route = self._call_map_api('route', params)

        # Adjust for weather conditions
        route = self._adjust_for_weather(route, weather_data)

        # Adjust for traffic conditions
        route = self._adjust_for_traffic(route, traffic_data)

        return route

    def get_traffic_data(self, location: Dict[str, float]) -> Dict:
        """Get traffic conditions for a location in Aruba"""
        params = {
            'location': f"{location['latitude']},{location['longitude']}",
            'radius': '5000',  # 5km radius
            'key': self.api_key
        }
        return self._call_map_api('traffic', params)

    def get_weather_data(self, location: Dict[str, float]) -> Dict:
        """Get weather conditions for a location in Aruba"""
        params = {
            'q': f"{location['latitude']},{location['longitude']}",
            'key': self.api_key
        }
        return self._call_map_api('weather', params)
    
    def _call_map_api(self, endpoint_type: str, params: Dict) -> Dict:
        """Call external mapping API with error handling"""
        endpoint = self.map_config.get(f"{endpoint_type}_endpoint")
        if not endpoint:
            raise ConfigurationError('foodie_map', f"{endpoint_type}_endpoint")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Map API error: {str(e)}")
            # Fallback to simulated data if API fails
            return self._simulate_api_data(endpoint_type, params)
    
    def _simulate_api_data(self, endpoint_type: str, params: Dict) -> Dict:
        """Generate simulated API data for development/testing"""
        logger.warning("Using simulated map data")
        
        if endpoint_type == 'route':
            return {
                "routes": [{
                    "summary": "Simulated Route",
                    "legs": [{
                        "distance": {"text": "5.2 km", "value": 5200},
                        "duration": {"text": "15 mins", "value": 900},
                        "duration_in_traffic": {"text": "18 mins", "value": 1080},
                        "steps": [
                            {"html_instructions": "Head north on Smith Blvd"},
                            {"html_instructions": "Turn right onto Palm Beach Rd"}
                        ]
                    }]
                }]
            }
        elif endpoint_type == 'traffic':
            return {
                "traffic_level": "moderate",
                "incidents": [
                    {"description": "Construction on Palm Beach Rd"}
                ]
            }
        elif endpoint_type == 'weather':
            return {
                "temp_c": 30,
                "condition": "Partly cloudy",
                "precip_mm": 0,
                "wind_kph": 25
            }
        return {}
    
    def _adjust_for_weather(self, route: Dict, weather: Dict) -> Dict:
        """Adjust route estimates based on weather conditions"""
        weather_factor = 1.0
        
        # Rain adjustment
        if weather.get('precip_mm', 0) > 5:
            weather_factor *= 1.3
            route['weather_warning'] = "Heavy rain expected"
        
        # Wind adjustment (especially important for scooters/bikes)
        if weather.get('wind_kph', 0) > 30:
            weather_factor *= 1.2
            route['weather_warning'] = "High winds expected"
        
        # Apply adjustments
        for leg in route.get('routes', [{}])[0].get('legs', []):
            if 'duration' in leg and 'value' in leg['duration']:
                leg['duration']['value'] = int(leg['duration']['value'] * weather_factor)
                leg['duration']['text'] = self._format_duration(leg['duration']['value'])
            
            if 'duration_in_traffic' in leg and 'value' in leg['duration_in_traffic']:
                leg['duration_in_traffic']['value'] = int(leg['duration_in_traffic']['value'] * weather_factor)
                leg['duration_in_traffic']['text'] = self._format_duration(leg['duration_in_traffic']['value'])
        
        return route
    
    def _adjust_for_traffic(self, route: Dict, traffic: Dict) -> Dict:
        """Adjust route based on traffic conditions"""
        traffic_factor = 1.0
        
        if traffic.get('traffic_level') == 'heavy':
            traffic_factor = 1.4
        elif traffic.get('traffic_level') == 'moderate':
            traffic_factor = 1.2
        
        # Apply adjustments
        for leg in route.get('routes', [{}])[0].get('legs', []):
            if 'duration_in_traffic' in leg and 'value' in leg['duration_in_traffic']:
                leg['duration_in_traffic']['value'] = int(leg['duration_in_traffic']['value'] * traffic_factor)
                leg['duration_in_traffic']['text'] = self._format_duration(leg['duration_in_traffic']['value'])
        
        return route
    
    def _format_duration(self, seconds: int) -> str:
        """Convert seconds to human-readable duration"""
        minutes = math.ceil(seconds / 60)
        if minutes < 60:
            return f"{minutes} mins"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}min"
    
    def _validate_coordinates(self, coords: Dict[str, float]):
        """Validate geographic coordinates"""
        lat = coords.get('latitude')
        lng = coords.get('longitude')
        
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise GeolocationError(
                "coordinate_validation", 
                (lat, lng)
            )
    
    def _check_service_area(self, location: Dict[str, float]):
        """Verify location is within Aruba service area"""
        aruba_bounds = {
            'min_lat': 12.41, 'max_lat': 12.63,
            'min_lng': -70.06, 'max_lng': -69.87
        }
        
        lat = location['latitude']
        lng = location['longitude']
        
        if not (aruba_bounds['min_lat'] <= lat <= aruba_bounds['max_lat'] and
                aruba_bounds['min_lng'] <= lng <= aruba_bounds['max_lng']):
            raise ServiceAreaError(
                f"{lat},{lng}", 
                "Aruba"
            )
            
    def calculate_eta_adjustment(self, route, traffic_data, weather_data):
        """Calculate time adjustment based on current conditions"""
        base_duration = route['routes'][0]['legs'][0]['duration']['value']
        
        # Calculate traffic factor
        traffic_factor = 1.0
        if traffic_data.get('traffic_level') == 'heavy':
            traffic_factor = 1.4
        elif traffic_data.get('traffic_level') == 'moderate':
            traffic_factor = 1.2
            
        # Calculate weather factor
        weather_factor = 1.0
        if weather_data.get('precip_mm', 0) > 5:
            weather_factor *= 1.3
        if weather_data.get('wind_kph', 0) > 30:
            weather_factor *= 1.2
            
        # Calculate adjusted duration
        adjusted_duration = base_duration * traffic_factor * weather_factor
        return adjusted_duration
    
    def calculate_distance(
        self, 
        point1: Dict[str, float], 
        point2: Dict[str, float]
    ) -> float:
        """
        Calculate Haversine distance between two points (in kilometers)
        """
        # Earth radius in kilometers
        R = 6371.0
        
        lat1 = math.radians(point1['latitude'])
        lon1 = math.radians(point1['longitude'])
        lat2 = math.radians(point2['latitude'])
        lon2 = math.radians(point2['longitude'])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def optimize_delivery_route(
        self, 
        start: Dict[str, float], 
        deliveries: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Optimize delivery route using nearest neighbor algorithm
        """
        if not deliveries:
            return []
        
        # Start from central hub if no start provided
        start = start or self.central_hub
        
        # Create a copy of deliveries to mutate
        unvisited = deliveries.copy()
        route = [start]
        current_point = start
        
        while unvisited:
            # Find nearest unvisited delivery point
            nearest = min(
                unvisited,
                key=lambda p: self.calculate_distance(current_point, p)
            )
            
            # Add to route and remove from unvisited
            route.append(nearest)
            unvisited.remove(nearest)
            current_point = nearest
        
        # Return to start point
        route.append(start)
        
        return route

if __name__ == "__main__":
    printer.status("MAIN", "Testing FoodieMap Methods", "info")
    fm = FoodieMap()
    
    # Test locations in Aruba
    locations = {
        "central_hub": fm.central_hub,
        "palm_beach": {"latitude": 12.564, "longitude": -70.043},
        "oranjestad": {"latitude": 12.518, "longitude": -70.037},
        "savaneta": {"latitude": 12.450, "longitude": -69.950}
    }
    
    # 1. Test distance calculation
    printer.status("TEST 1", "Test distance calculation", "info")
    distance = fm.calculate_distance(
        locations['central_hub'], 
        locations['palm_beach']
    )
    printer.pretty("DISTANCE (hub to Palm Beach)", f"{distance:.2f} km", "info")
    
    # 2. Test route optimization
    printer.status("TEST 2", "Test route optimization", "info")
    deliveries = [
        locations['palm_beach'],
        locations['oranjestad'],
        locations['savaneta']
    ]
    optimized_route = fm.optimize_delivery_route(
        locations['central_hub'], 
        deliveries
    )
    printer.pretty("OPTIMIZED ROUTE", optimized_route, "info")
    
    # 3. Test full route planning
    printer.status("TEST 3", "Test full route planning", "info")
    route = fm.get_route(
        origin=locations['central_hub'],
        destination=locations['savaneta'],
        waypoints=[locations['palm_beach'], locations['oranjestad']]
    )
    printer.pretty("FULL ROUTE WITH TRAFFIC/WEATHER", route, "success")
    
    # 4. Test service area validation
    printer.status("TEST 4", "Test service area validation", "info")
    try:
        # This should be outside Aruba
        fm._check_service_area({"latitude": 12.30, "longitude": -70.00})
    except ServiceAreaError as e:
        printer.pretty("SERVICE AREA VALIDATION", str(e), "warning")