# Application Roadmap — Progressive Contracts

From library-with-viewer to application. Each contract builds on the previous.

## Tier 1 — "Makes it feel like an application"

### APP-01: In-place parameter editing with live regeneration
**Status**: Pending
**Depends on**: None
**Scope**: Side panel form fields for constellation parameters (altitude, inclination, planes, sats/plane, phase factor, RAAN offset) become editable after layer creation. Changing a value triggers layer regeneration via existing `update_layer()` / `recompute_analysis()` backend. Analysis layers re-derive from updated source constellation.

**Acceptance criteria**:
- Walker layer shows editable parameter fields in the layer list
- Changing altitude regenerates constellation + dependent analysis layers
- CelesTrak layers show read-only params (not user-editable)
- Parameters persist across session save/load
- No full page reload — CZML data source swapped in-place

### APP-02: Metrics summary panel
**Status**: Pending
**Depends on**: None
**Scope**: Quantitative metrics extracted from analysis computations and displayed below each analysis layer in the side panel. Backend returns metrics alongside CZML. Frontend renders as compact key-value pairs.

**Acceptance criteria**:
- Coverage analysis shows: avg visible sats, min visible sats, % area with N+ coverage
- Eclipse analysis shows: avg % time sunlit, max continuous eclipse duration
- Conjunction hazard shows: number of close approaches, closest approach distance
- Deorbit compliance shows: N compliant / N total, worst-case lifetime
- Station-keeping shows: avg annual delta-V, max annual delta-V
- Beta angle shows: min/max/avg beta angle across constellation
- Metrics appear in layer list panel below each analysis layer
- Metrics included in session save/load
- API endpoint returns metrics in layer state response

### APP-03: Satellite data table
**Status**: Pending
**Depends on**: None
**Scope**: Collapsible table panel in the viewer showing per-satellite data for the selected constellation. Columns: name, plane, altitude (km), inclination (deg), RAAN (deg), period (min), beta angle (deg), eclipse %. Sortable by any column. Click row to fly camera to satellite.

**Acceptance criteria**:
- Table panel toggleable via side panel button or keyboard shortcut
- Shows data for selected/active constellation layer
- Columns sortable by click on header
- Click row flies Cesium camera to that satellite entity
- Table updates when constellation changes (edit params, switch layer)
- Backend API endpoint provides tabular satellite data
- Works for both Walker and CelesTrak constellations

### APP-04: Named scenarios with description
**Status**: Pending
**Depends on**: None
**Scope**: Session save prompts for scenario name and optional description. Load dialog shows saved scenario metadata. Recent scenarios list in side panel.

**Acceptance criteria**:
- Save dialog prompts for name (required) and description (optional)
- Saved JSON includes name, description, timestamp, layer summary
- Load shows scenario name, description, date, layer count before loading
- Side panel shows "Recent" section with last 5 saved scenario names
- Recent list stored in localStorage
- Scenario file includes version field for forward compatibility

## Tier 2 — "Makes it recommendable"

### APP-05: Parameter sweep / trade study
**Status**: Pending
**Depends on**: APP-01 (editable params), APP-02 (metrics)
**Scope**: Sweep one or two constellation parameters across a range, compute metrics for each configuration, display results as a chart. Backend runs batch computations. Frontend renders Chart.js line/scatter plot.

**Acceptance criteria**:
- UI: select parameter to sweep, min/max/step, select metric to plot
- Supports single-param sweep (line chart) and dual-param sweep (heatmap)
- Backend endpoint accepts sweep definition, returns array of {params, metrics}
- Progress indication during sweep computation
- Chart embedded in viewer (Chart.js or similar, no external deps)
- Sweep results exportable as CSV
- Cancellable (abort long sweeps)

### APP-06: Side-by-side configuration comparison
**Status**: Pending
**Depends on**: APP-02 (metrics), APP-03 (data table)
**Scope**: Compare two constellation configurations on the same metrics. Split view or overlay with delta indicators. Metrics table shows absolute values and differences.

**Acceptance criteria**:
- UI: select two constellation layers to compare
- Metrics comparison table: Config A | Config B | Delta
- Visual overlay mode: both constellations visible with distinct colors
- Data table can switch between configurations
- Works with any two constellation layers (Walker vs Walker, Walker vs CelesTrak)

### APP-07: Constraint definition and pass/fail reporting
**Status**: Pending
**Depends on**: APP-02 (metrics)
**Scope**: Define requirements as metric thresholds (e.g., min coverage > 85%, max revisit < 60 min). Run analysis, get green/red pass/fail per constraint. Summary badge on constellation layer.

**Acceptance criteria**:
- UI: define constraints as {metric, operator, threshold} tuples
- Constraint set saved with scenario
- Analysis results evaluated against constraints automatically
- Pass/fail badges displayed on layer items (green check / red X)
- Summary: "5/7 constraints met" with expandable detail
- Constraints exportable as part of session JSON
- Pre-built constraint templates (e.g., "ITU deorbit compliance", "LEO coverage baseline")

## Tier 3 — "Makes it competitive"

### APP-08: Report generation
**Status**: Pending
**Depends on**: APP-02 (metrics), APP-07 (constraints)
**Scope**: Generate HTML report with constellation parameters, analysis figures (globe screenshots via Cesium), metrics tables, constraint pass/fail, and export as downloadable file.

**Acceptance criteria**:
- One-click "Generate Report" button in side panel
- Report includes: scenario name/description, constellation parameters, 3D globe snapshot, metrics tables per analysis, constraint results, timestamp
- Output as self-contained HTML file (printable to PDF via browser)
- Report styled professionally (no framework branding)
- Includes all active layers and their metrics

### APP-09: CLI batch mode for trade studies
**Status**: Pending
**Depends on**: APP-05 (sweep logic)
**Scope**: CLI command for running parameter sweeps without the viewer. Output as CSV or JSON. Reuses backend sweep logic from APP-05.

**Acceptance criteria**:
- `humeris sweep --param altitude_km:400:800:50 --metric coverage_pct --output results.csv`
- Supports multiple --param flags for multi-dimensional sweeps
- Supports multiple --metric flags
- Output formats: CSV (default), JSON (--format json)
- Progress bar on stderr
- Exit code 0 on success, 1 on error
- Can combine with --constraint for pass/fail filtering

### APP-10: CCSDS OEM/OPM import
**Status**: Pending
**Depends on**: None
**Scope**: Import spacecraft state from CCSDS Orbit Ephemeris Message (OEM) and Orbit Parameter Message (OPM) files. Standard interchange format used by GMAT, STK, and operational systems.

**Acceptance criteria**:
- Parse CCSDS OEM (KVN format) into OrbitalState list
- Parse CCSDS OPM (KVN format) into OrbitalState
- CLI flag: `--import-oem file.oem`, `--import-opm file.opm`
- Viewer: drag-and-drop or file upload for OEM/OPM
- Supports ECI (GCRF) and ECF (ITRF) reference frames
- Handles single and multi-segment OEM files
- Validation: reject malformed files with clear error message
- Domain parser in domain layer (stdlib only, no external deps)

## Execution Order

```
APP-01 → APP-02 → APP-03 → APP-04          (Tier 1, parallel-safe)
    ↓        ↓        ↓
APP-05 → APP-06 → APP-07                    (Tier 2, builds on Tier 1)
    ↓                 ↓
APP-09          APP-08                       (Tier 3)
                                APP-10       (Tier 3, independent)
```

Tier 1 items are independent of each other and can be implemented in any order.
Tier 2 items depend on Tier 1 metrics and editing capabilities.
Tier 3 items depend on Tier 2 sweep and constraint logic (except APP-10).
