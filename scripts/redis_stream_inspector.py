#!/usr/bin/env python3
"""
scripts/redis_stream_inspector.py

Advanced Redis stream inspection and debugging tool.
Provides real-time monitoring, analysis, and troubleshooting capabilities.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis


class RedisStreamInspector:
    """
    Comprehensive Redis stream inspection tool.

    Features:
    - Stream monitoring and analysis
    - Message inspection and decoding
    - Performance metrics
    - Data validation
    - Export capabilities
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis stream inspector.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
        """
        self.host = host
        self.port = port
        self.db = db

        # Connect to Redis
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.client.ping()
            print(f"✅ Connected to Redis at {host}:{port}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            sys.exit(1)

        # Statistics
        self.stats = defaultdict(
            lambda: {
                "message_count": 0,
                "total_bytes": 0,
                "first_seen": None,
                "last_seen": None,
                "message_rate": 0.0,
            }
        )

    def list_streams(self) -> List[str]:
        """List all Redis streams."""
        try:
            # Get all keys
            keys = self.client.keys("*")

            # Filter for streams
            streams = []
            for key in keys:
                try:
                    key_type = self.client.type(key)
                    if key_type == b"stream":
                        streams.append(key.decode("utf-8"))
                except Exception:
                    continue

            return sorted(streams)

        except Exception as e:
            print(f"Error listing streams: {e}")
            return []

    def get_stream_info(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a stream."""
        try:
            info = self.client.xinfo_stream(stream_name)

            # Convert bytes to strings for readability
            readable_info = {}
            for key, value in info.items():
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                if isinstance(value, bytes):
                    readable_info[key_str] = value.decode("utf-8")
                elif isinstance(value, list):
                    # Handle nested structures
                    readable_info[key_str] = self._convert_bytes_recursive(value)
                else:
                    readable_info[key_str] = value

            return readable_info

        except Exception as e:
            print(f"Error getting stream info: {e}")
            return None

    def _convert_bytes_recursive(self, obj):
        """Recursively convert bytes to strings in nested structures."""
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except Exception:
                return str(obj)
        elif isinstance(obj, list):
            return [self._convert_bytes_recursive(item) for item in obj]
        elif isinstance(obj, dict):
            return {self._convert_bytes_recursive(k): self._convert_bytes_recursive(v) for k, v in obj.items()}
        else:
            return obj

    def read_stream(
        self,
        stream_name: str,
        count: int = 10,
        start_id: str = "-",
        decode_images: bool = False,
    ) -> List[Tuple[str, Dict]]:
        """
        Read messages from a stream.

        Args:
            stream_name: Name of the stream
            count: Number of messages to read
            start_id: Starting message ID ("-" for beginning)
            decode_images: Whether to decode base64 image data

        Returns:
            List of (message_id, data) tuples
        """
        try:
            messages = self.client.xrange(stream_name, min=start_id, max="+", count=count)

            results = []
            for msg_id, data in messages:
                msg_id_str = msg_id.decode("utf-8")

                # Decode message data
                decoded_data = {}
                for key, value in data.items():
                    key_str = key.decode("utf-8")

                    # Try to decode as string
                    try:
                        value_str = value.decode("utf-8")

                        # Try to parse JSON
                        try:
                            decoded_data[key_str] = json.loads(value_str)
                        except Exception:
                            # Check if it's base64 image data
                            if decode_images and key_str == "image_data":
                                decoded_data[key_str] = "<base64 image data>"
                            else:
                                decoded_data[key_str] = value_str
                    except Exception:
                        decoded_data[key_str] = f"<binary data: {len(value)} bytes>"

                results.append((msg_id_str, decoded_data))

            return results

        except Exception as e:
            print(f"Error reading stream: {e}")
            return []

    def monitor_stream(
        self,
        stream_name: str,
        interval: float = 1.0,
        max_duration: Optional[float] = None,
    ):
        """
        Monitor a stream in real-time.

        Args:
            stream_name: Name of the stream to monitor
            interval: Polling interval in seconds
            max_duration: Maximum monitoring duration (None for indefinite)
        """
        print(f"\n📡 Monitoring stream: {stream_name}")
        print(f"Polling interval: {interval}s")
        print("Press Ctrl+C to stop")
        print("─" * 70)

        last_id = "$"  # Start with latest
        start_time = time.time()
        message_count = 0

        try:
            while True:
                # Check duration limit
                if max_duration and (time.time() - start_time) > max_duration:
                    break

                # Read new messages
                try:
                    messages = self.client.xread({stream_name: last_id}, count=10, block=int(interval * 1000))

                    if messages:
                        for stream, msgs in messages:
                            for msg_id, data in msgs:
                                message_count += 1
                                last_id = msg_id.decode("utf-8")

                                # Display message
                                timestamp = datetime.fromtimestamp(int(msg_id.decode("utf-8").split("-")[0]) / 1000)
                                print(f"\n[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] Message #{message_count}")
                                print(f"ID: {last_id}")

                                # Display fields
                                for key, value in data.items():
                                    key_str = key.decode("utf-8")

                                    try:
                                        value_str = value.decode("utf-8")

                                        # Truncate long values
                                        if len(value_str) > 100:
                                            value_str = value_str[:100] + "..."

                                        # Try to parse JSON for better formatting
                                        try:
                                            value_obj = json.loads(value_str)
                                            print(f"  {key_str}: {json.dumps(value_obj, indent=2)}")
                                        except Exception:
                                            print(f"  {key_str}: {value_str}")
                                    except Exception:
                                        print(f"  {key_str}: <binary: {len(value)} bytes>")

                except redis.exceptions.ResponseError as e:
                    print(f"Redis error: {e}")
                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped")
            print(f"Total messages received: {message_count}")

    def analyze_stream(self, stream_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """
        Analyze stream contents and structure.

        Args:
            stream_name: Name of the stream
            sample_size: Number of messages to analyze

        Returns:
            Analysis results dictionary
        """
        print(f"\n🔍 Analyzing stream: {stream_name}")
        print(f"Sample size: {sample_size}")

        messages = self.read_stream(stream_name, count=sample_size)

        if not messages:
            print("No messages found")
            return {}

        # Analyze structure
        field_types = defaultdict(set)
        field_sizes = defaultdict(list)
        timestamps = []

        for msg_id, data in messages:
            # Extract timestamp from message ID
            ts = int(msg_id.split("-")[0])
            timestamps.append(ts)

            # Analyze fields
            for key, value in data.items():
                # Determine type
                if isinstance(value, dict):
                    field_types[key].add("object")
                elif isinstance(value, list):
                    field_types[key].add("array")
                elif isinstance(value, (int, float)):
                    field_types[key].add("number")
                elif isinstance(value, str):
                    if value.startswith("<"):
                        field_types[key].add("binary")
                    else:
                        field_types[key].add("string")
                else:
                    field_types[key].add(type(value).__name__)

                # Track size
                if isinstance(value, str):
                    field_sizes[key].append(len(value))

        # Calculate statistics
        analysis = {
            "stream_name": stream_name,
            "sample_size": len(messages),
            "fields": {},
            "timing": {},
        }

        # Field analysis
        for field, types in field_types.items():
            avg_size = np.mean(field_sizes[field]) if field_sizes[field] else 0
            max_size = max(field_sizes[field]) if field_sizes[field] else 0

            analysis["fields"][field] = {
                "types": list(types),
                "avg_size": int(avg_size),
                "max_size": int(max_size),
            }

        # Timing analysis
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            analysis["timing"] = {
                "avg_interval_ms": float(np.mean(intervals)),
                "min_interval_ms": float(np.min(intervals)),
                "max_interval_ms": float(np.max(intervals)),
                "rate_hz": 1000.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0,
            }

        return analysis

    def print_analysis(self, analysis: Dict[str, Any]):
        """Pretty-print analysis results."""
        print("\n" + "=" * 70)
        print("Stream Analysis Results")
        print("=" * 70)

        print(f"\nStream: {analysis['stream_name']}")
        print(f"Sample size: {analysis['sample_size']} messages")

        # Fields
        print("\nFields:")
        for field, info in analysis["fields"].items():
            print(f"\n  {field}:")
            print(f"    Types: {', '.join(info['types'])}")
            if info["avg_size"] > 0:
                print(f"    Avg size: {info['avg_size']} bytes")
                print(f"    Max size: {info['max_size']} bytes")

        # Timing
        if "timing" in analysis:
            timing = analysis["timing"]
            print("\nTiming:")
            print(f"  Message rate: {timing['rate_hz']:.2f} Hz")
            print(f"  Avg interval: {timing['avg_interval_ms']:.2f} ms")
            print(f"  Min interval: {timing['min_interval_ms']:.2f} ms")
            print(f"  Max interval: {timing['max_interval_ms']:.2f} ms")

        print("\n" + "=" * 70)

    def compare_streams(self, stream1: str, stream2: str, sample_size: int = 50):
        """Compare two streams side-by-side."""
        print("\n🔄 Comparing streams:")
        print(f"  Stream 1: {stream1}")
        print(f"  Stream 2: {stream2}")

        analysis1 = self.analyze_stream(stream1, sample_size)
        analysis2 = self.analyze_stream(stream2, sample_size)

        print("\n" + "=" * 70)
        print("Stream Comparison")
        print("=" * 70)

        # Compare fields
        fields1 = set(analysis1.get("fields", {}).keys())
        fields2 = set(analysis2.get("fields", {}).keys())

        common_fields = fields1 & fields2
        unique1 = fields1 - fields2
        unique2 = fields2 - fields1

        print(f"\nCommon fields ({len(common_fields)}): {', '.join(sorted(common_fields))}")
        if unique1:
            print(f"Only in {stream1}: {', '.join(sorted(unique1))}")
        if unique2:
            print(f"Only in {stream2}: {', '.join(sorted(unique2))}")

        # Compare timing
        if "timing" in analysis1 and "timing" in analysis2:
            print("\nMessage Rates:")
            print(f"  {stream1}: {analysis1['timing']['rate_hz']:.2f} Hz")
            print(f"  {stream2}: {analysis2['timing']['rate_hz']:.2f} Hz")

        print("\n" + "=" * 70)

    def export_stream(
        self,
        stream_name: str,
        output_file: str,
        format: str = "json",
        max_messages: int = 1000,
    ):
        """
        Export stream data to file.

        Args:
            stream_name: Stream to export
            output_file: Output file path
            format: Export format ("json" or "csv")
            max_messages: Maximum messages to export
        """
        print(f"\n💾 Exporting stream: {stream_name}")
        print(f"Format: {format}")
        print(f"Output: {output_file}")

        messages = self.read_stream(stream_name, count=max_messages)

        if not messages:
            print("No messages to export")
            return

        try:
            if format == "json":
                # Export as JSON
                export_data = [
                    {
                        "id": msg_id,
                        "data": data,
                    }
                    for msg_id, data in messages
                ]

                with open(output_file, "w") as f:
                    json.dump(export_data, f, indent=2)

            elif format == "csv":
                import csv

                # Determine all fields
                all_fields = set()
                for _, data in messages:
                    all_fields.update(data.keys())

                # Write CSV
                with open(output_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["id"] + sorted(all_fields))
                    writer.writeheader()

                    for msg_id, data in messages:
                        row = {"id": msg_id}
                        row.update(data)
                        writer.writerow(row)

            else:
                print(f"Unknown format: {format}")
                return

            print(f"✅ Exported {len(messages)} messages to {output_file}")

        except Exception as e:
            print(f"❌ Export failed: {e}")

    def interactive_mode(self):
        """Run in interactive command-line mode."""
        print("\n" + "=" * 70)
        print("Redis Stream Inspector - Interactive Mode")
        print("=" * 70)

        while True:
            print("\nCommands:")
            print("  1. List streams")
            print("  2. Stream info")
            print("  3. Read messages")
            print("  4. Monitor stream")
            print("  5. Analyze stream")
            print("  6. Compare streams")
            print("  7. Export stream")
            print("  q. Quit")

            choice = input("\nEnter command: ").strip()

            if choice == "q":
                break

            elif choice == "1":
                streams = self.list_streams()
                print(f"\nFound {len(streams)} streams:")
                for stream in streams:
                    info = self.get_stream_info(stream)
                    length = info.get("length", "?") if info else "?"
                    print(f"  - {stream} ({length} messages)")

            elif choice == "2":
                stream = input("Stream name: ").strip()
                info = self.get_stream_info(stream)
                if info:
                    print(f"\nStream: {stream}")
                    for key, value in info.items():
                        print(f"  {key}: {value}")

            elif choice == "3":
                stream = input("Stream name: ").strip()
                count = int(input("Message count (default 10): ").strip() or "10")
                messages = self.read_stream(stream, count=count)

                print(f"\nRead {len(messages)} messages:")
                for msg_id, data in messages:
                    print(f"\n[{msg_id}]")
                    for key, value in data.items():
                        print(f"  {key}: {value}")

            elif choice == "4":
                stream = input("Stream name: ").strip()
                interval = float(input("Polling interval (s, default 1.0): ").strip() or "1.0")
                self.monitor_stream(stream, interval=interval)

            elif choice == "5":
                stream = input("Stream name: ").strip()
                sample_size = int(input("Sample size (default 100): ").strip() or "100")
                analysis = self.analyze_stream(stream, sample_size)
                self.print_analysis(analysis)

            elif choice == "6":
                stream1 = input("First stream: ").strip()
                stream2 = input("Second stream: ").strip()
                sample_size = int(input("Sample size (default 50): ").strip() or "50")
                self.compare_streams(stream1, stream2, sample_size)

            elif choice == "7":
                stream = input("Stream name: ").strip()
                output = input("Output file: ").strip()
                format = input("Format (json/csv, default json): ").strip() or "json"
                max_msgs = int(input("Max messages (default 1000): ").strip() or "1000")
                self.export_stream(stream, output, format, max_msgs)

            else:
                print("Invalid command")


# # List all streams
# python scripts/redis_stream_inspector.py list
#
# # Get stream info
# python scripts/redis_stream_inspector.py info robot_camera
#
# # Read recent messages
# python scripts/redis_stream_inspector.py read detected_objects --count 20
#
# # Monitor in real-time
# python scripts/redis_stream_inspector.py monitor robot_camera --interval 0.5
#
# # Analyze stream structure
# python scripts/redis_stream_inspector.py analyze detected_objects --sample 100
#
# # Compare two streams
# python scripts/redis_stream_inspector.py compare robot_camera annotated_camera
#
# # Export to JSON
# python scripts/redis_stream_inspector.py export detected_objects output.json
#
# # Interactive mode (explore all features)
# python scripts/redis_stream_inspector.py interactive


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Redis Stream Inspector and Debugging Tool")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--db", type=int, default=0, help="Redis database")

    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List all streams")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get stream info")
    info_parser.add_argument("stream", help="Stream name")

    # Read command
    read_parser = subparsers.add_parser("read", help="Read messages")
    read_parser.add_argument("stream", help="Stream name")
    read_parser.add_argument("--count", type=int, default=10, help="Message count")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor stream")
    monitor_parser.add_argument("stream", help="Stream name")
    monitor_parser.add_argument("--interval", type=float, default=1.0, help="Polling interval")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze stream")
    analyze_parser.add_argument("stream", help="Stream name")
    analyze_parser.add_argument("--sample", type=int, default=100, help="Sample size")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare streams")
    compare_parser.add_argument("stream1", help="First stream")
    compare_parser.add_argument("stream2", help="Second stream")
    compare_parser.add_argument("--sample", type=int, default=50, help="Sample size")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export stream")
    export_parser.add_argument("stream", help="Stream name")
    export_parser.add_argument("output", help="Output file")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json")
    export_parser.add_argument("--max-messages", type=int, default=1000)

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")

    args = parser.parse_args()

    # Create inspector
    inspector = RedisStreamInspector(host=args.host, port=args.port, db=args.db)

    # Execute command
    if args.command == "list":
        streams = inspector.list_streams()
        print(f"\nFound {len(streams)} streams:")
        for stream in streams:
            info = inspector.get_stream_info(stream)
            length = info.get("length", "?") if info else "?"
            print(f"  - {stream} ({length} messages)")

    elif args.command == "info":
        info = inspector.get_stream_info(args.stream)
        if info:
            print(f"\nStream: {args.stream}")
            for key, value in info.items():
                print(f"  {key}: {value}")

    elif args.command == "read":
        messages = inspector.read_stream(args.stream, count=args.count)
        print(f"\nRead {len(messages)} messages:")
        for msg_id, data in messages:
            print(f"\n[{msg_id}]")
            for key, value in data.items():
                print(f"  {key}: {value}")

    elif args.command == "monitor":
        inspector.monitor_stream(args.stream, interval=args.interval)

    elif args.command == "analyze":
        analysis = inspector.analyze_stream(args.stream, sample_size=args.sample)
        inspector.print_analysis(analysis)

    elif args.command == "compare":
        inspector.compare_streams(args.stream1, args.stream2, sample_size=args.sample)

    elif args.command == "export":
        inspector.export_stream(
            args.stream,
            args.output,
            format=args.format,
            max_messages=args.max_messages,
        )

    elif args.command == "interactive" or args.command is None:
        inspector.interactive_mode()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
