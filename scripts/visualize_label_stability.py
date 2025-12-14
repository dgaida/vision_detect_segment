#!/usr/bin/env python3
"""
scripts/visualize_label_stability.py

Real-time visualization of label stabilization for tracked objects.
Shows how labels evolve from detection to stabilization.
"""

import sys
import time
import numpy as np
from typing import Dict, Optional
from collections import Counter
from redis_robot_comm import RedisMessageBroker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


class LabelStabilityVisualizer:
    """
    Visualizes label stabilization process for tracked objects.

    Features:
    - Real-time tracking of label history
    - Visualization of label confidence evolution
    - Detection of stabilization events
    - Live plotting of label distribution
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        detection_stream: str = "detected_objects",
        stabilization_frames: int = 10,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.detection_stream = detection_stream
        self.stabilization_frames = stabilization_frames

        # Connect to Redis
        try:
            self.broker = RedisMessageBroker(redis_host, redis_port)
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            sys.exit(1)

        # Tracking data
        self.track_data: Dict[int, Dict] = {}
        self.frame_count = 0
        self.running = True

        # For matplotlib animation
        self.fig = None
        self.axes = None

    def process_detections(self):
        """Process incoming detections from Redis."""
        print("📊 Monitoring detections for label stability...")
        print(f"Stream: {self.detection_stream}")
        print(f"Stabilization after: {self.stabilization_frames} frames")
        print("─" * 70)

        consecutive_failures = 0
        max_failures = 10

        while self.running:
            try:
                # Get latest detections
                result = self.broker.get_latest_objects(timeout_seconds=0.5)

                if result:
                    objects, metadata = result
                    self.frame_count += 1

                    # Process each detected object
                    for obj in objects:
                        if "track_id" in obj:
                            self._update_track_data(obj)

                    consecutive_failures = 0
                else:
                    # No data available
                    time.sleep(0.1)

            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"\n❌ Redis error after {consecutive_failures} attempts: {e}")
                    break
                time.sleep(0.5)

    def _update_track_data(self, obj: Dict):
        """Update tracking data for an object."""
        track_id = obj["track_id"]
        label = obj["label"]

        # Initialize track data if new
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "label_history": [],
                "frame_count": 0,
                "first_seen": self.frame_count,
                "last_seen": self.frame_count,
                "is_stabilized": False,
                "stabilized_label": None,
                "stabilization_frame": None,
            }

        track = self.track_data[track_id]

        # Update tracking info
        track["label_history"].append(label)
        track["frame_count"] += 1
        track["last_seen"] = self.frame_count

        # Check for stabilization
        if not track["is_stabilized"] and track["frame_count"] >= self.stabilization_frames:
            # Label has stabilized
            label_counter = Counter(track["label_history"])
            most_common = label_counter.most_common(1)[0][0]

            track["is_stabilized"] = True
            track["stabilized_label"] = most_common
            track["stabilization_frame"] = self.frame_count

            print(f"\n🔒 Track ID {track_id} STABILIZED")
            print(f"   Label: '{most_common}'")
            print(f"   After {track['frame_count']} frames")
            print(f"   History: {dict(label_counter)}")

    def get_track_summary(self, track_id: int) -> Optional[Dict]:
        """Get summary information for a track."""
        if track_id not in self.track_data:
            return None

        track = self.track_data[track_id]
        label_counter = Counter(track["label_history"])

        # Calculate label stability score (0-1)
        if len(track["label_history"]) > 0:
            most_common_count = label_counter.most_common(1)[0][1]
            stability = most_common_count / len(track["label_history"])
        else:
            stability = 0.0

        return {
            "track_id": track_id,
            "frame_count": track["frame_count"],
            "first_seen": track["first_seen"],
            "last_seen": track["last_seen"],
            "is_stabilized": track["is_stabilized"],
            "stabilized_label": track.get("stabilized_label"),
            "stabilization_frame": track.get("stabilization_frame"),
            "label_distribution": dict(label_counter),
            "stability_score": stability,
            "unique_labels": len(label_counter),
        }

    def print_summary(self):
        """Print summary of all tracked objects."""
        print("\n" + "=" * 70)
        print("Label Stability Summary")
        print("=" * 70)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Active tracks: {len(self.track_data)}")
        print()

        for track_id in sorted(self.track_data.keys()):
            summary = self.get_track_summary(track_id)
            if summary:
                status = "✅ STABLE" if summary["is_stabilized"] else "⏳ PENDING"
                print(f"Track #{track_id:3d} | {status}")
                print(f"  Frames: {summary['frame_count']:3d} | " f"Stability: {summary['stability_score']:.2%}")

                if summary["is_stabilized"]:
                    print(
                        f"  Label: '{summary['stabilized_label']}' " f"(stabilized at frame {summary['stabilization_frame']})"
                    )
                else:
                    # Show current majority
                    dist = summary["label_distribution"]
                    majority = max(dist, key=dist.get)
                    print(f"  Current majority: '{majority}'")

                # Show label distribution
                print(f"  Distribution: {summary['label_distribution']}")
                print()

        print("=" * 70)

    def visualize_interactive(self):
        """Create interactive matplotlib visualization."""
        print("\n📊 Starting interactive visualization...")
        print("Close the plot window to stop monitoring")

        # Create figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle("Label Stability Visualization", fontsize=16, fontweight="bold")

        # Start processing in background thread
        process_thread = threading.Thread(target=self.process_detections, daemon=True)
        process_thread.start()

        # Animation function
        def update_plot(frame):
            if not self.running or len(self.track_data) == 0:
                return

            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()

            # Plot 1: Label stability timeline
            ax1 = self.axes[0, 0]
            self._plot_stability_timeline(ax1)

            # Plot 2: Label distribution per track
            ax2 = self.axes[0, 1]
            self._plot_label_distribution(ax2)

            # Plot 3: Stabilization progress
            ax3 = self.axes[1, 0]
            self._plot_stabilization_progress(ax3)

            # Plot 4: Track statistics
            ax4 = self.axes[1, 1]
            self._plot_track_statistics(ax4)

            self.fig.tight_layout()

        # Create animation
        FuncAnimation(self.fig, update_plot, interval=500, cache_frame_data=False)

        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("\n🛑 Stopping visualization...")

    def _plot_stability_timeline(self, ax):
        """Plot label changes over time for each track."""
        ax.set_title("Label History Timeline")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Track ID")

        if not self.track_data:
            ax.text(0.5, 0.5, "Waiting for data...", ha="center", va="center")
            return

        # Plot each track's label history
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        label_colors = {}

        for track_id, track in self.track_data.items():
            history = track["label_history"]
            first_frame = track["first_seen"]

            # Assign colors to unique labels
            for label in set(history):
                if label not in label_colors:
                    label_colors[label] = colors[len(label_colors) % 10]

            # Plot label changes
            for i, label in enumerate(history):
                frame = first_frame + i
                ax.scatter(
                    frame,
                    track_id,
                    c=[label_colors[label]],
                    s=50,
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=0.5,
                )

            # Mark stabilization point
            if track["is_stabilized"]:
                stab_frame = track["stabilization_frame"]
                ax.axvline(stab_frame, color="green", linestyle="--", alpha=0.3)
                ax.scatter(stab_frame, track_id, marker="*", s=200, c="green", zorder=5)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
            for label, color in label_colors.items()
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        ax.grid(True, alpha=0.3)

    def _plot_label_distribution(self, ax):
        """Plot label distribution for each active track."""
        ax.set_title("Label Distribution per Track")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")

        if not self.track_data:
            ax.text(0.5, 0.5, "Waiting for data...", ha="center", va="center")
            return

        # Collect all unique labels
        all_labels = set()
        for track in self.track_data.values():
            all_labels.update(track["label_history"])

        all_labels = sorted(all_labels)

        # Create grouped bar chart
        x = np.arange(len(all_labels))
        width = 0.8 / max(1, len(self.track_data))

        for i, (track_id, track) in enumerate(self.track_data.items()):
            counter = Counter(track["label_history"])
            counts = [counter.get(label, 0) for label in all_labels]

            color = "green" if track["is_stabilized"] else "orange"
            ax.bar(
                x + i * width,
                counts,
                width,
                label=f"Track {track_id}",
                alpha=0.7,
                color=color,
            )

        ax.set_xticks(x + width * (len(self.track_data) - 1) / 2)
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    def _plot_stabilization_progress(self, ax):
        """Plot progress toward stabilization for each track."""
        ax.set_title("Stabilization Progress")
        ax.set_xlabel("Track ID")
        ax.set_ylabel("Frames / Stability Score")

        if not self.track_data:
            ax.text(0.5, 0.5, "Waiting for data...", ha="center", va="center")
            return

        track_ids = sorted(self.track_data.keys())
        frame_counts = []
        stability_scores = []

        for track_id in track_ids:
            summary = self.get_track_summary(track_id)
            frame_counts.append(summary["frame_count"])
            stability_scores.append(summary["stability_score"] * 100)

        x = np.arange(len(track_ids))
        width = 0.35

        # Bar chart for frame count
        bars1 = ax.bar(
            x - width / 2,
            frame_counts,
            width,
            label="Frame Count",
            alpha=0.7,
        )

        # Add stabilization threshold line
        ax.axhline(
            self.stabilization_frames,
            color="red",
            linestyle="--",
            label="Stabilization Threshold",
        )

        # Color bars based on stabilization
        for i, track_id in enumerate(track_ids):
            if self.track_data[track_id]["is_stabilized"]:
                bars1[i].set_color("green")
            else:
                bars1[i].set_color("orange")

        # Twin axis for stability score
        ax2 = ax.twinx()
        ax2.plot(
            x,
            stability_scores,
            "bo-",
            label="Stability Score",
            linewidth=2,
            markersize=8,
        )
        ax2.set_ylabel("Stability Score (%)", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        ax2.set_ylim(0, 105)

        ax.set_xticks(x)
        ax.set_xticklabels([f"#{tid}" for tid in track_ids])
        ax.set_xlabel("Track ID")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _plot_track_statistics(self, ax):
        """Plot overall track statistics."""
        ax.set_title("Track Statistics")
        ax.axis("off")

        if not self.track_data:
            ax.text(0.5, 0.5, "Waiting for data...", ha="center", va="center")
            return

        # Calculate statistics
        total_tracks = len(self.track_data)
        stabilized_tracks = sum(1 for t in self.track_data.values() if t["is_stabilized"])
        avg_frames = np.mean([t["frame_count"] for t in self.track_data.values()])
        avg_stability = np.mean([self.get_track_summary(tid)["stability_score"] for tid in self.track_data.keys()])

        # Create text summary
        stats_text = f"""
        Total Tracks: {total_tracks}
        Stabilized: {stabilized_tracks} ({stabilized_tracks/max(1,total_tracks)*100:.1f}%)
        Pending: {total_tracks - stabilized_tracks}

        Avg Frames per Track: {avg_frames:.1f}
        Avg Stability Score: {avg_stability:.1%}

        Total Frames: {self.frame_count}
        Stabilization Threshold: {self.stabilization_frames} frames
        """

        ax.text(
            0.1,
            0.9,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def run_console_mode(self):
        """Run in console-only mode without matplotlib."""
        print("\n📊 Starting console monitoring...")
        print("Press Ctrl+C to stop")
        print("─" * 70)

        try:
            self.process_detections()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
        finally:
            self.running = False
            self.print_summary()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize label stability for tracked objects")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--stream", default="detected_objects", help="Detection stream name")
    parser.add_argument(
        "--stabilization-frames",
        type=int,
        default=10,
        help="Frames required for stabilization",
    )
    parser.add_argument(
        "--console-only",
        action="store_true",
        help="Run in console mode without GUI",
    )

    args = parser.parse_args()

    visualizer = LabelStabilityVisualizer(
        redis_host=args.host,
        redis_port=args.port,
        detection_stream=args.stream,
        stabilization_frames=args.stabilization_frames,
    )

    try:
        if args.console_only:
            visualizer.run_console_mode()
        else:
            visualizer.visualize_interactive()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    finally:
        visualizer.print_summary()


if __name__ == "__main__":
    main()
