# Copyright (c) Jeroen Visser. MIT License.
"""Humeris GUI — Satellite Constellation Export Tool.

A standalone graphical tool for exporting constellations to 9 supported
formats. Uses tkinter (stdlib). Designed to be operable by a 12-year-old.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionSpec:
    """A format-specific option (checkbox or spin)."""

    key: str
    label: str
    type: str  # "bool" or "int"
    default: Any
    exporter_kwarg: str


@dataclass(frozen=True)
class FormatSpec:
    """Specification for one export format."""

    key: str
    label: str
    description: str
    extension: str
    default_filename: str
    exporter_factory: Callable[..., Any]
    options: list[OptionSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exporter factories — thin wrappers around adapter constructors
# ---------------------------------------------------------------------------

def _csv_factory(**_kwargs: Any) -> Any:
    from humeris.adapters.csv_exporter import CsvSatelliteExporter
    return CsvSatelliteExporter()


def _geojson_factory(**_kwargs: Any) -> Any:
    from humeris.adapters.geojson_exporter import GeoJsonSatelliteExporter
    return GeoJsonSatelliteExporter()


def _kml_factory(**kwargs: Any) -> Any:
    from humeris.adapters.kml_exporter import KmlExporter
    return KmlExporter(**kwargs)


def _celestia_factory(**_kwargs: Any) -> Any:
    from humeris.adapters.celestia_exporter import CelestiaExporter
    return CelestiaExporter()


def _stellarium_factory(**_kwargs: Any) -> Any:
    from humeris.adapters.stellarium_exporter import StellariumExporter
    return StellariumExporter()


def _blender_factory(**kwargs: Any) -> Any:
    from humeris.adapters.blender_exporter import BlenderExporter
    return BlenderExporter(**kwargs)


def _spaceengine_factory(**_kwargs: Any) -> Any:
    from humeris.adapters.spaceengine_exporter import SpaceEngineExporter
    return SpaceEngineExporter()


def _ksp_factory(**kwargs: Any) -> Any:
    from humeris.adapters.ksp_exporter import KspExporter
    return KspExporter(**kwargs)


def _ubox_factory(**_kwargs: Any) -> Any:
    from humeris.adapters.ubox_exporter import UboxExporter
    return UboxExporter()


# ---------------------------------------------------------------------------
# Format specifications
# ---------------------------------------------------------------------------

FORMAT_SPECS: list[FormatSpec] = [
    FormatSpec(
        key="csv",
        label="CSV",
        description="Spreadsheet data (.csv)",
        extension=".csv",
        default_filename="constellation.csv",
        exporter_factory=_csv_factory,
        options=[],
    ),
    FormatSpec(
        key="geojson",
        label="GeoJSON",
        description="For mapping tools (.geojson)",
        extension=".geojson",
        default_filename="constellation.geojson",
        exporter_factory=_geojson_factory,
        options=[],
    ),
    FormatSpec(
        key="kml",
        label="KML",
        description="For Google Earth (.kml)",
        extension=".kml",
        default_filename="constellation.kml",
        exporter_factory=_kml_factory,
        options=[
            OptionSpec("include_orbits", "Show orbit lines", "bool", True, "include_orbits"),
            OptionSpec("include_planes", "Group by plane", "bool", False, "include_planes"),
            OptionSpec("include_isl", "Show links between satellites", "bool", False, "include_isl"),
        ],
    ),
    FormatSpec(
        key="celestia",
        label="Celestia",
        description="Celestia planetarium (.ssc)",
        extension=".ssc",
        default_filename="constellation.ssc",
        exporter_factory=_celestia_factory,
        options=[],
    ),
    FormatSpec(
        key="stellarium",
        label="Stellarium",
        description="Planetarium app (.tle)",
        extension=".tle",
        default_filename="constellation.tle",
        exporter_factory=_stellarium_factory,
        options=[],
    ),
    FormatSpec(
        key="blender",
        label="Blender",
        description="3D modeling script (.py)",
        extension=".py",
        default_filename="constellation.py",
        exporter_factory=_blender_factory,
        options=[
            OptionSpec("include_orbits", "Show orbit lines", "bool", True, "include_orbits"),
            OptionSpec("color_by_plane", "Color by plane", "bool", False, "color_by_plane"),
        ],
    ),
    FormatSpec(
        key="spaceengine",
        label="SpaceEngine",
        description="SpaceEngine catalog (.sc)",
        extension=".sc",
        default_filename="constellation.sc",
        exporter_factory=_spaceengine_factory,
        options=[],
    ),
    FormatSpec(
        key="ksp",
        label="KSP",
        description="Kerbal Space Program (.sfs)",
        extension=".sfs",
        default_filename="constellation.sfs",
        exporter_factory=_ksp_factory,
        options=[
            OptionSpec("scale_to_kerbin", "Scale orbits to Kerbin", "bool", True, "scale_to_kerbin"),
        ],
    ),
    FormatSpec(
        key="ubox",
        label="Universe Sandbox",
        description="Universe Sandbox simulation (.ubox)",
        extension=".ubox",
        default_filename="constellation.ubox",
        exporter_factory=_ubox_factory,
        options=[],
    ),
]

# ---------------------------------------------------------------------------
# CelesTrak groups
# ---------------------------------------------------------------------------

CELESTRAK_GROUPS: list[str] = [
    "STARLINK",
    "ONEWEB",
    "STATIONS",
    "ACTIVE",
    "VISUAL",
    "WEATHER",
    "NOAA",
    "GOES",
    "RESOURCE",
    "SARSAT",
    "GPS-OPS",
    "GALILEO",
    "BEIDOU",
    "IRIDIUM",
    "IRIDIUM-NEXT",
    "GLOBALSTAR",
    "ORBCOMM",
    "AMATEUR",
    "SCIENCE",
    "GEODETIC",
]


# ---------------------------------------------------------------------------
# Satellite loading
# ---------------------------------------------------------------------------

def load_default_satellites() -> list[Any]:
    """Load the default constellation (Walker shells + SSO band)."""
    from humeris.cli import get_default_shells
    from humeris.domain.constellation import generate_walker_shell

    satellites: list[Any] = []
    for shell in get_default_shells():
        satellites.extend(generate_walker_shell(shell))
    return satellites


# ---------------------------------------------------------------------------
# Export orchestration
# ---------------------------------------------------------------------------

def run_export(
    satellites: list[Any],
    spec: FormatSpec,
    output_dir: str,
    filename: str,
    options: dict[str, Any],
) -> int:
    """Run a single export. Returns number of satellites exported."""
    # Map option keys to exporter kwargs
    factory_kwargs: dict[str, Any] = {}
    for opt in spec.options:
        if opt.key in options:
            factory_kwargs[opt.exporter_kwarg] = options[opt.key]
        else:
            factory_kwargs[opt.exporter_kwarg] = opt.default

    exporter = spec.exporter_factory(**factory_kwargs)
    path = os.path.join(output_dir, filename)
    return exporter.export(satellites, path)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class HumerisGui:
    """Main GUI window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Humeris — Satellite Constellation Export")
        self.root.geometry("620x780")
        self.root.resizable(True, True)

        self._satellites: list[Any] = []
        self._format_vars: dict[str, tk.BooleanVar] = {}
        self._filename_vars: dict[str, tk.StringVar] = {}
        self._option_vars: dict[str, dict[str, tk.BooleanVar]] = {}
        self._option_frames: dict[str, tk.Frame] = {}
        self._source_var = tk.StringVar(value="default")
        self._celestrak_group_var = tk.StringVar(value="STARLINK")
        self._status_var = tk.StringVar(value="Loading default constellation...")
        self._export_status_var = tk.StringVar(value="")

        # Default output directory
        docs = Path.home() / "Documents"
        desktop = Path.home() / "Desktop"
        if docs.exists():
            default_dir = str(docs / "humeris-export")
        elif desktop.exists():
            default_dir = str(desktop / "humeris-export")
        else:
            default_dir = str(Path.home() / "humeris-export")
        self._output_dir_var = tk.StringVar(value=default_dir)

        self._build_ui()
        self._load_defaults_async()

    def _build_ui(self) -> None:
        """Build the complete UI."""
        # Main scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event: Any) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll_up(event: Any) -> None:
            canvas.yview_scroll(-1, "units")

        def _on_linux_scroll_down(event: Any) -> None:
            canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_linux_scroll_up)
        canvas.bind_all("<Button-5>", _on_linux_scroll_down)

        pad = {"padx": 10, "pady": 5}

        # --- Source section ---
        src_frame = ttk.LabelFrame(scroll_frame, text="WHERE DO THE SATELLITES COME FROM?")
        src_frame.pack(fill="x", **pad)

        ttk.Radiobutton(
            src_frame,
            text="Default constellation (4,752 satellites)",
            variable=self._source_var,
            value="default",
            command=self._on_source_change,
        ).pack(anchor="w", padx=5, pady=2)

        cel_frame = ttk.Frame(src_frame)
        cel_frame.pack(fill="x", padx=5, pady=2)
        ttk.Radiobutton(
            cel_frame,
            text="Live from CelesTrak:",
            variable=self._source_var,
            value="celestrak",
            command=self._on_source_change,
        ).pack(side="left")
        self._celestrak_combo = ttk.Combobox(
            cel_frame,
            textvariable=self._celestrak_group_var,
            values=CELESTRAK_GROUPS,
            state="readonly",
            width=20,
        )
        self._celestrak_combo.pack(side="left", padx=5)
        self._celestrak_combo.bind("<<ComboboxSelected>>", lambda _: self._on_source_change())

        ttk.Label(src_frame, textvariable=self._status_var).pack(anchor="w", padx=5, pady=2)

        # --- Format section ---
        fmt_frame = ttk.LabelFrame(scroll_frame, text="PICK YOUR EXPORT FORMATS")
        fmt_frame.pack(fill="x", **pad)

        for spec in FORMAT_SPECS:
            self._build_format_row(fmt_frame, spec)

        # --- Output directory ---
        dir_frame = ttk.LabelFrame(scroll_frame, text="SAVE TO")
        dir_frame.pack(fill="x", **pad)

        dir_row = ttk.Frame(dir_frame)
        dir_row.pack(fill="x", padx=5, pady=5)
        ttk.Entry(dir_row, textvariable=self._output_dir_var, width=50).pack(side="left", fill="x", expand=True)
        ttk.Button(dir_row, text="Pick", command=self._pick_directory).pack(side="left", padx=5)

        # --- Export button ---
        btn_frame = ttk.Frame(scroll_frame)
        btn_frame.pack(fill="x", **pad)

        self._export_btn = ttk.Button(
            btn_frame,
            text="EXPORT!",
            command=self._on_export,
        )
        self._export_btn.pack(fill="x", padx=20, pady=10, ipady=8)

        ttk.Label(btn_frame, textvariable=self._export_status_var).pack(anchor="center", pady=5)

    def _build_format_row(self, parent: ttk.Frame, spec: FormatSpec) -> None:
        """Build a single format row with checkbox, filename, and options."""
        var = tk.BooleanVar(value=(spec.key == "csv"))
        self._format_vars[spec.key] = var

        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=2)

        ttk.Checkbutton(
            row,
            text=f"{spec.label} — {spec.description}",
            variable=var,
            command=lambda k=spec.key: self._on_format_toggle(k),
        ).pack(anchor="w")

        # Filename entry + browse
        file_frame = ttk.Frame(row)
        file_frame.pack(fill="x", padx=20, pady=1)

        fn_var = tk.StringVar(value=spec.default_filename)
        self._filename_vars[spec.key] = fn_var
        ttk.Entry(file_frame, textvariable=fn_var, width=35).pack(side="left")
        ttk.Button(
            file_frame,
            text="Browse",
            command=lambda k=spec.key, ext=spec.extension: self._browse_filename(k, ext),
        ).pack(side="left", padx=5)

        # Format-specific options
        if spec.options:
            opt_frame = ttk.Frame(row)
            opt_frame.pack(fill="x", padx=20, pady=1)
            self._option_frames[spec.key] = opt_frame
            self._option_vars[spec.key] = {}

            for opt in spec.options:
                opt_var = tk.BooleanVar(value=opt.default)
                self._option_vars[spec.key][opt.key] = opt_var
                ttk.Checkbutton(opt_frame, text=opt.label, variable=opt_var).pack(
                    side="left", padx=5,
                )

    def _on_format_toggle(self, key: str) -> None:
        """Show/hide options when format is toggled."""
        if key in self._option_frames:
            if self._format_vars[key].get():
                self._option_frames[key].pack(fill="x", padx=20, pady=1)
            else:
                self._option_frames[key].pack_forget()

    def _pick_directory(self) -> None:
        """Open folder picker for output directory."""
        d = filedialog.askdirectory(
            title="Select output folder",
            initialdir=self._output_dir_var.get(),
        )
        if d:
            self._output_dir_var.set(d)

    def _browse_filename(self, key: str, ext: str) -> None:
        """Open file save dialog for a specific format."""
        f = filedialog.asksaveasfilename(
            title=f"Save {key} file",
            defaultextension=ext,
            initialfile=self._filename_vars[key].get(),
            initialdir=self._output_dir_var.get(),
        )
        if f:
            self._filename_vars[key].set(os.path.basename(f))
            self._output_dir_var.set(os.path.dirname(f))

    def _on_source_change(self) -> None:
        """Handle source radio button change."""
        if self._source_var.get() == "default":
            self._load_defaults_async()
        else:
            self._fetch_celestrak_async()

    def _load_defaults_async(self) -> None:
        """Load default constellation in background thread."""
        self._status_var.set("Loading default constellation...")

        def _load() -> None:
            try:
                sats = load_default_satellites()
                self.root.after(0, lambda: self._on_satellites_loaded(sats))
            except Exception as e:
                self.root.after(0, lambda: self._status_var.set(f"Error: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def _fetch_celestrak_async(self) -> None:
        """Fetch satellites from CelesTrak in background thread."""
        group = self._celestrak_group_var.get()
        self._status_var.set(f"Fetching {group} from CelesTrak...")

        def _fetch() -> None:
            try:
                from humeris.adapters.celestrak import CelesTrakAdapter
                adapter = CelesTrakAdapter()
                sats = adapter.fetch_satellites(group=group.lower())
                self.root.after(0, lambda: self._on_satellites_loaded(sats))
            except ImportError:
                self.root.after(
                    0,
                    lambda: self._status_var.set(
                        "CelesTrak requires sgp4: pip install humeris-core[live]"
                    ),
                )
            except Exception as e:
                self.root.after(0, lambda: self._status_var.set(f"Error: {e}"))

        threading.Thread(target=_fetch, daemon=True).start()

    def _on_satellites_loaded(self, satellites: list[Any]) -> None:
        """Callback when satellites are loaded."""
        self._satellites = satellites
        count = len(satellites)
        self._status_var.set(f"{count:,} satellites ready")

    def _on_export(self) -> None:
        """Run export for all checked formats."""
        if not self._satellites:
            messagebox.showwarning("No satellites", "No satellites loaded yet. Please wait.")
            return

        selected = [s for s in FORMAT_SPECS if self._format_vars[s.key].get()]
        if not selected:
            messagebox.showwarning("No formats", "Please select at least one export format.")
            return

        output_dir = self._output_dir_var.get()
        os.makedirs(output_dir, exist_ok=True)

        self._export_btn.configure(state="disabled")
        self._export_status_var.set("Exporting...")

        def _do_export() -> None:
            results: list[str] = []
            errors: list[str] = []

            for spec in selected:
                filename = self._filename_vars[spec.key].get()
                options: dict[str, Any] = {}
                if spec.key in self._option_vars:
                    for opt_key, opt_var in self._option_vars[spec.key].items():
                        options[opt_key] = opt_var.get()

                try:
                    count = run_export(self._satellites, spec, output_dir, filename, options)
                    results.append(f"{spec.label}: {count:,} satellites → {filename}")
                except Exception as e:
                    errors.append(f"{spec.label}: {e}")

            def _done() -> None:
                self._export_btn.configure(state="normal")
                if errors:
                    msg = "\n".join(results + [""] + ["ERRORS:"] + errors)
                    self._export_status_var.set(f"Exported with {len(errors)} error(s)")
                    messagebox.showwarning("Export completed with errors", msg)
                else:
                    self._export_status_var.set(
                        f"Exported {len(self._satellites):,} satellites to {len(results)} file(s)"
                    )
                    if len(results) <= 5:
                        messagebox.showinfo("Export complete", "\n".join(results))

            self.root.after(0, _done)

        threading.Thread(target=_do_export, daemon=True).start()

    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console script entry point."""
    gui = HumerisGui()
    gui.run()


if __name__ == "__main__":
    main()
