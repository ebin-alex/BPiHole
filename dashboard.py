import logging
import threading
import time
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
import webbrowser
import json
import traceback
import signal
import sys
import os

class Dashboard:
    def __init__(self, host='0.0.0.0', port=5000):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.running = False
        self.detections = []
        self.stats = {
            'total_packets': 0,
            'blackhole_detections': 0,
            'last_update': int(time.time())
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Setup routes
        self.setup_routes()
        self.logger.info("Dashboard initialized")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal")
        self.stop()
        # Force exit after a short delay to ensure cleanup
        threading.Timer(1.0, lambda: os._exit(0)).start()
        
    def setup_routes(self):
        @self.app.route('/')
        def home():
            try:
                return render_template_string('''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Blackhole Detection Dashboard</title>
                        <style>
                            body { 
                                font-family: Arial, sans-serif; 
                                margin: 20px;
                                background-color: #f5f5f5;
                            }
                            .container {
                                max-width: 1200px;
                                margin: 0 auto;
                            }
                            .header {
                                background-color: #fff;
                                padding: 20px;
                                border-radius: 5px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                margin-bottom: 20px;
                            }
                            .stats {
                                display: grid;
                                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                                gap: 20px;
                                margin-bottom: 20px;
                            }
                            .stat-card {
                                background-color: #fff;
                                padding: 15px;
                                border-radius: 5px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            .detection { 
                                background-color: #fff;
                                border: 1px solid #ddd;
                                padding: 15px;
                                margin: 10px 0;
                                border-radius: 5px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                transition: all 0.3s ease;
                            }
                            .detection:hover {
                                transform: translateY(-2px);
                                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                            }
                            .blackhole {
                                background-color: #ffe6e6;
                                border-color: #ff9999;
                            }
                            .search-bar {
                                width: 100%;
                                padding: 10px;
                                margin-bottom: 20px;
                                border: 1px solid #ddd;
                                border-radius: 5px;
                            }
                            .error {
                                color: red;
                                padding: 10px;
                                margin: 10px 0;
                                background-color: #ffe6e6;
                                border-radius: 5px;
                            }
                            .status-indicator {
                                display: inline-block;
                                width: 10px;
                                height: 10px;
                                border-radius: 50%;
                                margin-right: 5px;
                            }
                            .status-active {
                                background-color: #4CAF50;
                            }
                            .status-inactive {
                                background-color: #f44336;
                            }
                        </style>
                        <script>
                            let searchTimeout;
                            let connectionStatus = true;
                            
                            function updateStats() {
                                fetch('/stats')
                                    .then(response => response.json())
                                    .then(data => {
                                        document.getElementById('total-packets').textContent = data.total_packets;
                                        document.getElementById('blackhole-detections').textContent = data.blackhole_detections;
                                        document.getElementById('last-update').textContent = new Date(data.last_update * 1000).toLocaleString();
                                        connectionStatus = true;
                                        updateConnectionStatus();
                                    })
                                    .catch(error => {
                                        console.error('Error updating stats:', error);
                                        showError('Failed to update statistics');
                                        connectionStatus = false;
                                        updateConnectionStatus();
                                    });
                            }
                            
                            function updateConnectionStatus() {
                                const indicator = document.getElementById('connection-status');
                                if (connectionStatus) {
                                    indicator.className = 'status-indicator status-active';
                                    indicator.title = 'Connected';
                                } else {
                                    indicator.className = 'status-indicator status-inactive';
                                    indicator.title = 'Disconnected';
                                }
                            }
                            
                            function updateDetections() {
                                const searchTerm = document.getElementById('search').value;
                                fetch(`/detections?search=${encodeURIComponent(searchTerm)}`)
                                    .then(response => response.json())
                                    .then(data => {
                                        const container = document.getElementById('detections');
                                        container.innerHTML = '';
                                        data.forEach(d => {
                                            const div = document.createElement('div');
                                            div.className = 'detection ' + (d.is_blackhole ? 'blackhole' : '');
                                            div.innerHTML = `
                                                <strong>Source IP:</strong> ${d.source_ip}<br>
                                                <strong>Destination IP:</strong> ${d.dest_ip}<br>
                                                <strong>Status:</strong> ${d.is_blackhole ? 'BLACKHOLE DETECTED' : 'Normal'}<br>
                                                <strong>Confidence:</strong> ${(d.confidence * 100).toFixed(2)}%<br>
                                                <strong>Time:</strong> ${new Date(d.timestamp * 1000).toLocaleString()}
                                            `;
                                            container.appendChild(div);
                                        });
                                    })
                                    .catch(error => {
                                        console.error('Error updating detections:', error);
                                        showError('Failed to update detections');
                                    });
                            }
                            
                            function showError(message) {
                                const errorDiv = document.createElement('div');
                                errorDiv.className = 'error';
                                errorDiv.textContent = message;
                                document.body.insertBefore(errorDiv, document.body.firstChild);
                                setTimeout(() => errorDiv.remove(), 5000);
                            }
                            
                            function handleSearch(event) {
                                clearTimeout(searchTimeout);
                                searchTimeout = setTimeout(updateDetections, 300);
                            }
                            
                            // Update every 2 seconds
                            setInterval(() => {
                                updateStats();
                                updateDetections();
                            }, 2000);
                            
                            // Initial update
                            updateStats();
                            updateDetections();
                        </script>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>Blackhole Detection Dashboard <span id="connection-status" class="status-indicator status-active" title="Connected"></span></h1>
                            </div>
                            
                            <div class="stats">
                                <div class="stat-card">
                                    <h3>Total Packets</h3>
                                    <p id="total-packets">0</p>
                                </div>
                                <div class="stat-card">
                                    <h3>Blackhole Detections</h3>
                                    <p id="blackhole-detections">0</p>
                                </div>
                                <div class="stat-card">
                                    <h3>Last Update</h3>
                                    <p id="last-update">Never</p>
                                </div>
                            </div>
                            
                            <input type="text" id="search" class="search-bar" placeholder="Search by IP address..." oninput="handleSearch(event)">
                            
                            <div id="detections"></div>
                        </div>
                    </body>
                    </html>
                ''')
            except Exception as e:
                self.logger.error(f"Error rendering home page: {str(e)}")
                return "Error loading dashboard", 500
            
        @self.app.route('/detections')
        def get_detections():
            try:
                search_term = request.args.get('search', '').lower()
                filtered_detections = self.detections[-50:]  # Get last 50 detections
                
                if search_term:
                    filtered_detections = [
                        d for d in filtered_detections
                        if search_term in d['source_ip'].lower() or 
                           search_term in d['dest_ip'].lower()
                    ]
                
                return jsonify(filtered_detections)
            except Exception as e:
                self.logger.error(f"Error getting detections: {str(e)}")
                return jsonify({"error": "Failed to get detections"}), 500
            
        @self.app.route('/stats')
        def get_stats():
            try:
                return jsonify(self.stats)
            except Exception as e:
                self.logger.error(f"Error getting stats: {str(e)}")
                return jsonify({"error": "Failed to get stats"}), 500
            
    def add_detection(self, detection):
        """Add a new detection to the dashboard"""
        try:
            self.detections.append(detection)
            # Keep only last 100 detections
            if len(self.detections) > 100:
                self.detections = self.detections[-100:]
                
            # Update stats
            self.stats['total_packets'] += 1
            if detection.get('is_blackhole', False):
                self.stats['blackhole_detections'] += 1
            self.stats['last_update'] = int(time.time())
            
            self.logger.debug(f"Added detection: {json.dumps(detection)}")
        except Exception as e:
            self.logger.error(f"Error adding detection: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def run(self):
        """Run the dashboard server"""
        try:
            self.logger.info(f"Starting dashboard on {self.host}:{self.port}")
            self.running = True
            
            # Open browser
            webbrowser.open(f'http://{self.host}:{self.port}')
            
            # Run Flask app in a separate thread
            server_thread = threading.Thread(target=self._run_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error running dashboard: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def _run_server(self):
        """Run the Flask server"""
        try:
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            self.logger.error(f"Error in server thread: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def stop(self):
        """Stop the dashboard server"""
        try:
            self.running = False
            self.logger.info("Dashboard stopped")
        except Exception as e:
            self.logger.error(f"Error stopping dashboard: {str(e)}")
            self.logger.error(traceback.format_exc()) 