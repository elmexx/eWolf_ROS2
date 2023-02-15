import rclpy
from rclpy.node import Node

import serial
import socket
import os
import sys

import reach_ros_node.driver

class ReachGPS(Node):

    # Set our parameters and the default socket to open
    def __init__(self,host,port):
        super().__init__('rech_gps_node')
        self.host = host
        self.port = port

    # Should open the connection and connect to the device
    # This will then also start publishing the information
    def start(self):
        # Try to connect to the device
        self.get_logger().info('Connecting to Reach RTK %s on port %s' % (str(self.host),str(self.port)))
        self.connect_to_device()
        try:
            # Create the driver
            driver = reach_ros_node.driver.RosNMEADriver()
            while not rclpy.ok():
                #GPS = soc.recv(1024)
                data = self.buffered_readLine().strip()
                # Debug print message line
                #print(data)
                # Try to parse this data!
                try:
                    driver.process_line(data)
                except ValueError as e:
                    self.get_logger().info("Value error, likely due to missing fields in the NMEA message. Error was: %s." % e)
        except:
            # Close GPS socket when done
            self.soc.close()



    # Try to connect to the device, allows for reconnection
    # Will loop till we get a connection, note we have a long timeout
    def connect_to_device(self):
        while not rclpy.ok():
            try:
                self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.soc.settimeout(5.0)
                self.soc.connect((self.host,self.port))
                self.get_logger().info('Successfully connected to device, starting publishing!')
                return
            except socket.timeout:
                self.get_logger().warning(30,'Socket connection timeout. Retrying...')
                continue
            except Exception as e:
                self.get_logger().error("Socket connection error. Error was: %s." % e)
                exit()


    # Try to connect to the device, assuming it just was disconnected
    # Will loop till we get a connection
    def reconnect_to_device(self):
        self.get_logger().warning('Device disconnected. Reconnecting...')
        self.soc.close()
        self.connect_to_device()



    # Read one line from the socket
    # We want to read a single line as this is a single nmea message
    # https://stackoverflow.com/a/41333900
    # Also set a timeout so we can make sure we have a valid socket
    # https://stackoverflow.com/a/15175067
    def buffered_readLine(self):
        line = ""
        while not rclpy.ok():
            # Try to get data from it
            try:
                part = self.soc.recv(1)
            except socket.timeout:
                self.reconnect_to_device()
                continue
            # See if we need to process the data
            if not part or len(part) == 0:
                self.reconnect_to_device()
                continue
            if part != "\n":
                line += part
            elif part == "\n":
                break
        return line




