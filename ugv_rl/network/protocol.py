import json
import socket
import struct

class NetworkProtocol:
    """
    Simple JSON-based protocol helper.
    Messages are prefixed with 4-byte length header.
    """
    
    @staticmethod
    def send_msg(sock, data):
        """Send a dictionary as JSON."""
        msg = json.dumps(data).encode('utf-8')
        # Prefix with length (Big Endian unsigned int)
        sock.sendall(struct.pack('>I', len(msg)) + msg)

    @staticmethod
    def recv_msg(sock):
        """Receive a JSON dictionary."""
        # Read length
        raw_msglen = NetworkProtocol._recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read data
        raw_data = NetworkProtocol._recvall(sock, msglen)
        if not raw_data:
            return None
        return json.loads(raw_data.decode('utf-8'))

    @staticmethod
    def _recvall(sock, n):
        """Helper to receive n bytes or return None if EOF is hit."""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
