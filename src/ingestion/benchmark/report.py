"""
BenchmarkReport: Generate formatted benchmark reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class BenchmarkReport:
    """Formatted benchmark report."""

    summary: Dict[str, Any]
    results_by_sample_size: Dict[int, Dict[str, Any]]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReport":
        """Create report from dictionary."""
        return cls(
            summary=data.get("summary", {}),
            results_by_sample_size=data.get("results_by_sample_size", {}),
            timestamp=data.get("timestamp", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "summary": self.summary,
            "results_by_sample_size": self.results_by_sample_size,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save report to file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "BenchmarkReport":
        """Load report from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def format_text(self) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 60,
            "INGESTION PIPELINE BENCHMARK REPORT",
            "=" * 60,
            f"Generated: {self.timestamp}",
            "",
        ]

        # Summary
        if self.summary:
            lines.extend([
                "SUMMARY",
                "-" * 40,
                f"  Speedup Factor: {self.summary.get('speedup_factor', 'N/A')}x",
                f"  Sequential:     {self.summary.get('sequential_imgs_per_sec', 'N/A')} img/s",
                f"  Parallel:       {self.summary.get('parallel_imgs_per_sec', 'N/A')} img/s",
                "",
            ])

        # Results by sample size
        for size, results in sorted(self.results_by_sample_size.items()):
            lines.extend([
                f"SAMPLE SIZE: {size}",
                "-" * 40,
            ])

            for mode in ["sequential", "parallel"]:
                if mode in results:
                    r = results[mode]
                    lines.extend([
                        f"  {mode.upper()}:",
                        f"    Batch Size:  {r.get('batch_size', 'N/A')}",
                        f"    Total Time:  {r.get('total_time_s', 'N/A')}s",
                        f"    Processed:   {r.get('processed', 'N/A')}",
                        f"    Skipped:     {r.get('skipped', 'N/A')}",
                        f"    Errors:      {r.get('errors', 'N/A')}",
                        f"    Speed:       {r.get('images_per_second', 'N/A')} img/s",
                    ])

                    # Stage breakdown
                    if "stage_times" in r:
                        lines.append("    Stage Times:")
                        for stage, time_s in r["stage_times"].items():
                            lines.append(f"      {stage}: {time_s}s")

            if "speedup" in results:
                lines.append(f"  SPEEDUP: {results['speedup']}x")

            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
