
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
from datetime import datetime

from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI Web Socket Manager")
printer = PrettyPrinter

class WebSocketManager:
    def __init__(self, order_id):
        self.config = load_global_config()
        self.socket_config = get_config_section('web_socket')
        self.status = self.socket_config.get('status')
        self.ws_url = self.socket_config.get('url')
        self.timeout = float(self.socket_config.get('timeout'))  # Default 5 seconds timeout

        self.channel_id = f"ws-{order_id}"
        self.created_at = datetime.utcnow().isoformat()

    def broadcast(self, message):
        """Broadcast message to WebSocket channel with error handling"""
        logger.info(f"Broadcasting to {self.channel_id}: {message}")
        
        if not self.ws_url:
            logger.error("WebSocket URL not configured. Broadcast aborted.")
            return
        
        try:
            # Establish WebSocket connection with timeout
            logger.debug(f"Connecting to WebSocket at {self.ws_url}")
            ws_conn = create_connection(
                self.ws_url,
                timeout=self.timeout
            )
            
            # Send message and log result
            ws_conn.send(message)
            logger.info(f"Message successfully sent to {self.channel_id}")
            
            # Gracefully close connection
            ws_conn.close()
            
        except WebSocketTimeoutException:
            logger.error(f"Connection timeout for {self.channel_id} ({self.timeout}s)")
        except WebSocketConnectionClosedException:
            logger.error(f"Connection closed unexpectedly for {self.channel_id}")
        except ConnectionRefusedError:
            logger.error(f"Connection refused for {self.ws_url}")
        except Exception as e:
            logger.exception(f"Unexpected error broadcasting to {self.channel_id}: {str(e)}")

    @staticmethod
    def create_channel(order_id):
        return WebSocketManager(order_id)