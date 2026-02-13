# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Universe Sandbox .ubox exporter.

Exports constellation satellites as a Universe Sandbox simulation file.
The .ubox format is a ZIP archive containing a simulation.json with
Earth and satellite body entities using ECI state vectors, plus
metadata files (version.ini, info.json, ui-state.json).

Reverse-engineered from Universe Sandbox Update 35.4.5 save files.

Optional enrichment with physical properties (mass, radius) from
DragConfig when provided.

No external dependencies — only stdlib json/zipfile/math/io.
"""
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from typing import Any

from humeris.domain.constellation import Satellite
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter
from humeris.adapters.enrichment import compute_satellite_enrichment


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_R_EARTH_M = OrbitalConstants.R_EARTH
_EARTH_MASS_KG = 5.97219e24
_EARTH_RADIUS_M = 6_371_000.0
_EARTH_ID = 3
_SATELLITE_BASE_ID = 100

# Default satellite mass (kg) and radius (m) when no DragConfig provided
_DEFAULT_SAT_MASS_KG = 500.0
_DEFAULT_SAT_RADIUS_M = 0.1


def _vec_str(x: float, y: float, z: float) -> str:
    """Format a 3-vector as Universe Sandbox semicolon-separated string."""
    return f"{x};{y};{z}"


def _build_earth_entity() -> dict[str, Any]:
    """Build a full Earth entity matching Universe Sandbox format.

    Includes Celestial, AppearanceComponent, CompositionComponent,
    TrailComponent, SurfaceGridComponent, and HeatComponent — all
    required for Universe Sandbox to render Earth with textures,
    atmosphere, clouds, city lights, and oceans.
    """
    return {
        "$type": "Body",
        "Name": "Earth",
        "Components": [
            {
                "SurfaceTemperatureOverride": 0,
                "AtmosphereMass": 5.89891208654881e+18,
                "MeanMolecularWeightDryAir": 28.97,
                "DegreesOfFreedom": 5,
                "AtmosphereHeightMultiplier": 1,
                "UseSimulatedEmissivity": True,
                "EmissivityIR": 0.78,
                "AtmosphereLayers": 1,
                "Luminosity": 0,
                "ColdStar": False,
                "Realistic": True,
                "CanBeRealistic": True,
                "StarType": 0,
                "Category": 3,
                "FluxFromStars": 0,
                "MagneticField": 0.31869,
                "MagPoleAngle": 12,
                "MagPoleAxis": "(0.00, 0.00, 0.00)",
                "SmoothLuminosityToIsochrone": False,
                "SmoothRadiusToIsochrone": False,
                "SmoothTemperatureToIsochrone": False,
                "SmoothLuminosityLastOutputValue": 0,
                "SmoothRadiusLastOutputValue": 0,
                "SmoothTemperatureLastOutputValue": 0,
                "$type": "Celestial",
            },
            {
                "PrefabSource": "",
                "ColorMapSource": "Textures/Planets/earth_diffuse",
                "IceMapSource": "",
                "HeightMapSource": "Textures/Planets/earth_height",
                "HeightMapSource2": "",
                "NormalMapSource": "Textures/Planets/earth_height_normals",
                "NormalMapSource2": "",
                "UseDiffuse": True,
                "EmissiveMapSource": "Textures/EarthNight_2500x1250Grids",
                "SpecularMapSource": "",
                "VegetationMapSource": "Textures/Planets/earth_vegetation",
                "UseNormals": True,
                "NormalMapStrength": 1,
                "UseHeightMap0": True,
                "HeightMapMix0": 1,
                "HeightMapOffset0": 0,
                "HeightMapFlipH0": False,
                "HeightMapFlipV0": False,
                "UseHeightMap1": False,
                "HeightMapMix1": 1,
                "HeightMapOffset1": 0,
                "HeightMapFlipH1": False,
                "HeightMapFlipV1": False,
                "LightColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
                "Tint": "RGBA(1.000, 1.000, 1.000, 1.000)",
                "Planet": {
                    "Colors": [
                        "RGBA(1.000, 1.000, 1.000, 1.000)",
                        "RGBA(0.500, 0.500, 0.500, 1.000)",
                        "RGBA(0.000, 0.000, 0.000, 1.000)",
                    ],
                    "originalColors": [
                        "RGBA(1.000, 1.000, 1.000, 1.000)",
                        "RGBA(0.500, 0.500, 0.500, 1.000)",
                        "RGBA(0.000, 0.000, 0.000, 1.000)",
                    ],
                    "customColors": [
                        "RGBA(1.000, 1.000, 1.000, 1.000)",
                        "RGBA(0.500, 0.500, 0.500, 1.000)",
                        "RGBA(0.000, 0.000, 0.000, 1.000)",
                    ],
                    "AtmosphereColorMode": 0,
                    "AtmosphereColor": "RGBA(0.212, 0.325, 0.510, 1.000)",
                    "originalAtmosphereColor": "RGBA(0.212, 0.325, 0.510, 1.000)",
                    "customAtmosphereColor": "RGBA(0.212, 0.325, 0.510, 1.000)",
                    "CloudColorMode": 0,
                    "CloudColor": "RGBA(0.996, 0.996, 0.996, 1.000)",
                    "CustomCloudAppearance": False,
                    "CloudOpacity": 1,
                    "CloudSetA": 1,
                    "CloudSetB": 4,
                    "originalCloudColor": "RGBA(0.996, 0.996, 0.996, 1.000)",
                    "customCloudColor": "RGBA(0.996, 0.996, 0.996, 1.000)",
                    "CloudCoverage": 0.9,
                    "CloudOpacitySimulationMode": 0,
                    "Contrast": 0,
                    "CityLightsColor": "RGBA(1.000, 0.600, 0.200, 1.000)",
                    "CityLightsColorMode": 0,
                    "originalCityLightsColor": "RGBA(1.000, 0.600, 0.200, 1.000)",
                    "customCityLightsColor": "RGBA(1.000, 0.600, 0.200, 1.000)",
                    "UseWater": True,
                    "WaterColor": "RGBA(0.000, 0.066, 0.184, 1.000)",
                    "WaterColorMode": 0,
                    "originalWaterColor": "RGBA(0.000, 0.066, 0.184, 1.000)",
                    "customWaterColor": "RGBA(0.000, 0.066, 0.184, 1.000)",
                    "UseIce": True,
                    "IceColor": "RGBA(0.920, 0.960, 0.980, 1.000)",
                    "IceColorMode": 0,
                    "originalIceColor": "RGBA(0.920, 0.960, 0.980, 1.000)",
                    "customIceColor": "RGBA(0.920, 0.960, 0.980, 1.000)",
                    "IceNoise": 0.5,
                    "IceOpacityNoise": 1,
                    "UseSnow": True,
                    "SnowColor": "RGBA(0.920, 0.960, 0.980, 1.000)",
                    "SnowColorMode": 0,
                    "originalSnowColor": "RGBA(0.920, 0.960, 0.980, 1.000)",
                    "customSnowColor": "RGBA(0.920, 0.960, 0.980, 1.000)",
                    "SnowNoise": 0.5,
                    "UseVegetation": True,
                    "VegetationColor": "RGBA(0.100, 0.300, 0.080, 1.000)",
                    "VegetationColorMode": 0,
                    "VegetationMode": 1,
                    "originalVegetationColor": "RGBA(0.100, 0.300, 0.080, 1.000)",
                    "customVegetationColor": "RGBA(0.100, 0.300, 0.080, 1.000)",
                    "ShowAtmosphere": True,
                    "ShowAtmosphereClouds": True,
                    "CloudSpeedSimulationMode": 0,
                    "cloudSpeedAtEquatorA": -12,
                    "cloudSpeedAtEquatorB": -10,
                    "bandRotationA": -0.000736073693129143,
                    "bandRotationB": 0.49939775788744,
                    "poleRotationA": 0.000736073693129143,
                    "poleRotationB": 0.50060224211256,
                    "AtmosphereSimulationMode": 0,
                    "customAtmosphereOpacity": 0.2,
                    "RayleighScatteringStrength": 1,
                    "RayleighSimulationMode": 0,
                    "DefaultRayleighScatteringStrength": 1,
                    "HazeType": 2,
                    "HeightmapIndex1": 0,
                    "HeightmapIndex2": 0,
                    "UseDynamicEmissive": True,
                    "CityLightMode": 0,
                    "CityLightSource": 0,
                    "CityLightSeed": 79824,
                    "VegetationHabitabilityMode": 0,
                    "CityLightsHabitibilityMode": 0,
                },
                "GasGiant": {
                    "Contrast": 0,
                    "Colors": [],
                    "UserChangedColors": False,
                    "originalColors": [],
                    "customColors": [],
                    "BandingOffsets": "0;0;0;0",
                },
                "BlackHole": {
                    "Color": "RGBA(0.000, 0.000, 0.000, 0.000)",
                    "ColorMode": 0,
                    "originalColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
                    "customColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
                },
                "Prefab": {
                    "Color": "RGBA(1.000, 1.000, 1.000, 1.000)",
                    "ColorMode": 0,
                    "originalColor": "RGBA(1.000, 1.000, 1.000, 1.000)",
                    "customColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
                    "Metallic": 0,
                    "Smoothness": 0.5,
                },
                "ShowSurfaceOnly": True,
                "$type": "AppearanceComponent",
            },
            {
                "targetRadius": 6371000,
                "SimulateRadius": True,
                "RadiusTuningFactor": 0.995768059933356,
                "depots": {
                    "Iron": {"Mass": 1.47388999097863e+24, "LockSurfaceTracking": False},
                    "Silicate": {"Mass": 4.49695032767345e+24, "LockSurfaceTracking": False},
                    "Argon": {"Mass": 4.76684676287693e+16, "LockSurfaceTracking": False},
                    "Oxygen": {"Mass": 1.07484518391847e+18, "LockSurfaceTracking": False},
                    "Carbon Dioxide": {"Mass": 2146727584260000.0, "LockSurfaceTracking": False},
                    "Water": {"Mass": 1.429662258654e+21, "LockSurfaceTracking": False},
                    "Nitrogen": {"Mass": 4.00329160016265e+18, "LockSurfaceTracking": False},
                },
                "$type": "CompositionComponent",
            },
            {
                "Opacity": 1,
                "Length": 1,
                "Mode": 0,
                "Positions": {"array": [], "full": False, "max": 4096, "index": 0},
                "$type": "TrailComponent",
            },
            {
                "ElevationToRadiusRatio": 0.003296186,
                "AtlasIndex": 0,
                "$type": "SurfaceGridComponent",
            },
            {
                "SurfaceTemperature": 287.0,
                "StartingTemperature": 287,
                "TemperatureInitialized": True,
                "Albedo": 0.306,
                "SurfaceHeatCapacity": 292681689.662109,
                "UserChangedSurfaceHeatCapacity": False,
                "OverrideStartingTemp": False,
                "BlackbodyColorMode": 0,
                "originalBlackbodyColor": "RGBA(0.000, 0.000, 0.000, 1.000)",
                "customBlackbodyColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
                "EmitsLight": True,
                "BlackbodyNoise": 1,
                "UseBlackbodyNoise": False,
                "$type": "HeatComponent",
            },
        ],
        "Id": _EARTH_ID,
        "HorizonID": "399",
        "Age": 1.42009200000002e+17,
        "Color": "RGBA(0.333, 0.396, 0.514, 1.000)",
        "DefaultGUIColor": "RGBA(0.333, 0.396, 0.514, 1.000)",
        "CustomColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
        "CustomGUIColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
        "UserChangedColor": False,
        "UserChangedGUIColor": False,
        "ColorPalette": 2,
        "PhysicsMass": _EARTH_MASS_KG,
        "Mass": _EARTH_MASS_KG,
        "Radius": _EARTH_RADIUS_M,
        "GravityRadius": 0,
        "Density": 5513.51271486863,
        "Generation": 0,
        "Flags": 371,
        "DisplayFlags": 3,
        "Orientation": "7.906715E-10;0.9973429;1.337984E-08;0.07284964",
        "AngularVelocity": "0;-7.29211533325724E-05;0",
        "RotationAxis": "0;-1;0",
        "Position": "0;0;0",
        "Velocity": "0;0;0",
        "Suspended": False,
        "LockPosition": False,
        "LockRotation": False,
        "ColorMode": 0,
        "GUIColorMode": 0,
        "Parent": -1,
        "Source": -1,
        "Group": 0,
        "CustomOrbitParentId": -1,
        "LockedProperties": [{"$v": "DisplayDensity"}, {"$v": "PericenterDist"}],
        "Origin": 0,
        "Category": "planet",
        "BudgetType": 0,
        "InMajorCollision": True,
        "NonSphericalGravityEnabled": False,
        "J2": 0,
        "DatabaseID": "9d66c712-d79a-4006-a3de-2dd9bb998556",
        "Description": None,
        "RelativeTo": 0,
    }


def _build_satellite_entity(
    sat: Satellite,
    entity_id: int,
    mass_kg: float,
    radius_m: float,
) -> dict[str, Any]:
    """Build a satellite body entity with ECI state vectors."""
    px, py, pz = sat.position_eci
    vx, vy, vz = sat.velocity_eci

    volume_m3 = (4.0 / 3.0) * math.pi * radius_m**3
    density = mass_kg / volume_m3 if volume_m3 > 0 else 0.0

    return {
        "$type": "Body",
        "Name": sat.name,
        "Components": [
            {
                "Type": 1,
                "Size": 1.17835,
                "Color2": "RGBA(0.000, 0.000, 0.000, 0.000)",
                "StartEnergy": 0,
                "Energy": 0,
                "Age": 0,
                "Seed": (entity_id * 7919) % 100000,
                "DoubleData": "127;129;103",
                "RandomOffset": "(0.00, 0.00, 0.00)",
                "AllowTransition": False,
                "Rotation": "(0.00000, 0.00000, 0.00000, 100000.00000)",
                "ShadingMode": 1,
                "DecayMode": 1,
                "originalColor": "RGBA(1.000, 1.000, 1.000, 1.000)",
                "Materials": {"Iron": mass_kg},
                "$type": "ParticleComponent",
            },
            {
                "SurfaceTemperature": 300,
                "StartingTemperature": 0,
                "TemperatureInitialized": True,
                "Albedo": 0,
                "SurfaceHeatCapacity": 40972507.0,
                "UserChangedSurfaceHeatCapacity": False,
                "OverrideStartingTemp": False,
                "BlackbodyColorMode": 1,
                "originalBlackbodyColor": "RGBA(0.000, 0.000, 0.000, 1.000)",
                "customBlackbodyColor": "RGBA(1.000, 1.000, 1.000, 1.000)",
                "EmitsLight": False,
                "BlackbodyNoise": 1,
                "UseBlackbodyNoise": False,
                "$type": "HeatComponent",
            },
        ],
        "Id": entity_id,
        "HorizonID": None,
        "Age": 0,
        "Color": "RGBA(1.000, 1.000, 1.000, 1.000)",
        "DefaultGUIColor": "RGBA(1.000, 1.000, 1.000, 1.000)",
        "CustomColor": "RGBA(0.000, 0.000, 0.000, 0.000)",
        "CustomGUIColor": "RGBA(1.000, 1.000, 1.000, 1.000)",
        "UserChangedColor": False,
        "UserChangedGUIColor": False,
        "ColorPalette": 2,
        "PhysicsMass": mass_kg,
        "Mass": mass_kg,
        "Radius": radius_m,
        "GravityRadius": 0,
        "Density": density,
        "Generation": 0,
        "Flags": 146,
        "DisplayFlags": 3,
        "Orientation": "0;0;0;1",
        "AngularVelocity": "0;0;0",
        "RotationAxis": "0;1;0",
        "Position": _vec_str(px, py, pz),
        "Velocity": _vec_str(vx, vy, vz),
        "Suspended": False,
        "LockPosition": False,
        "LockRotation": False,
        "ColorMode": 0,
        "GUIColorMode": 0,
        "Parent": _EARTH_ID,
        "Source": -1,
        "Group": 0,
        "CustomOrbitParentId": -1,
        "LockedProperties": [{"$v": "DisplayDensity"}],
        "Origin": 0,
        "Category": "",
        "BudgetType": 0,
        "InMajorCollision": False,
        "NonSphericalGravityEnabled": False,
        "J2": 0,
        "DatabaseID": "00000000-0000-0000-0000-000000000000",
        "Description": None,
        "RelativeTo": 1,
    }


def _build_simulation(
    satellites: list[Satellite],
    epoch: datetime,
    mass_kg: float,
    radius_m: float,
    name: str,
) -> dict[str, Any]:
    """Build the full simulation.json structure."""
    date_str = epoch.strftime("%Y-%m-%d %I:%M %p").lower()

    entities: list[dict[str, Any]] = [_build_earth_entity()]
    for i, sat in enumerate(satellites):
        entity_id = _SATELLITE_BASE_ID + i
        entity = _build_satellite_entity(sat, entity_id, mass_kg, radius_m)
        enrich = compute_satellite_enrichment(sat, epoch)
        entity["Description"] = (
            f"Altitude: {enrich.altitude_km:.1f} km | "
            f"Inclination: {enrich.inclination_deg:.2f} deg | "
            f"Period: {enrich.orbital_period_min:.2f} min | "
            f"Beta angle: {enrich.beta_angle_deg:.2f} deg | "
            f"Atm. density: {enrich.atmospheric_density_kg_m3:.3e} kg/m3 | "
            f"L-shell: {enrich.l_shell:.2f}"
        )
        entities.append(entity)

    return {
        "Settings": {
            "ProminenceDurationMultiplier": 1,
            "ProminenceFrequencyMultiplier": 1,
            "CoolingMultiplier": 1,
            "VolatileAccelerationMultiplier": 1,
            "ThermalDiffusivity": 4807692,
            "TidalHeatingMultiplier": 1,
            "BackgroundTemperatureIntensity": 1,
            "VolatilityMassMultiplier": 1,
            "VolatileSpeedMultiplier": 1,
            "PotentialGridScale": 1,
            "CameraAngle": "78.96;45.86;0",
            "CameraPosition": "-39701469;420900977;-101308784",
            "CameraTargetDistance": 428832965.0,
            "CameraTargetId": _EARTH_ID,
            "LightIntensity": 1,
            "ParticleDensity": 0,
            "CameraFOV": 45,
            "CameraAttachPoint": "-0.49;0.06;0.87",
            "OpacityMultiplier": 1,
            "MarkerSize": 24,
            "View": {
                "VisualBodyScaleMultiplier": 4,
                "MinimumParticleOpacity": 0.2,
                "SubscaleFade": 1,
                "ParticleScaleMultiplier": 4,
                "ParticleScaleStyle": 1,
                "BodyScaleMultiplier": 4,
                "BodyScaleStyle": 1,
                "Ev100Offset": 0,
                "RimLightIntensity": 1,
                "EnhancedStarSurfaceIntensity": 1,
                "EnhancedStarSurface": True,
                "ExaggeratedLightingFactor": 0.7,
                "BrightnessFilterValue": 0.5,
                "BrightnessFilterOption": 3,
                "BloomThreshold": 0.8,
                "BloomScatter": 0.7,
                "BloomIntensity": 2.5,
                "VisualSubscaleFade": 0.9999999,
                "VisualParticleScaleMultiplier": 4,
                "VisualBrightnessFilterValue": 0.5,
            },
        },
        "Ambience": {"Seed": 0, "AmbientColor": "0.01;0.01;0.01;1", "Mode": 0},
        "Name": name,
        "Description": None,
        "Date": date_str,
        "DateActive": False,
        "TargetTimeStepPerRealSec": 4800,
        "MaximalTimeStepPerRealSec": "Infinity",
        "Tolerance": 1,
        "TimePassed": 0,
        "Accuracy": 2.33333333333333,
        "IntegratorId": 8,
        "IntegrationMode": 1,
        "Gravity": 6.6740831e-11,
        "ConstantCenterOfMass": False,
        "Pause": True,
        "AdaptiveIntegration": True,
        "UserChangedAdaptiveIntegration": False,
        "NBodySystemTolerance": 1,
        "AutoSpeed": {},
        "AutoCamera": {},
        "Entities": entities,
    }


class UboxExporter(SatelliteExporter):
    """Exports satellites as a Universe Sandbox .ubox simulation file.

    Produces a ZIP archive containing simulation.json (Earth + satellite
    bodies with ECI state vectors), version.ini, info.json, and
    ui-state.json — matching the Universe Sandbox Update 35 format.

    When drag_config is provided, satellite mass and radius are derived
    from it. Otherwise defaults to 500 kg / 0.1 m.
    """

    def __init__(self, drag_config=None, name: str = "Constellation"):
        self._drag_config = drag_config
        self._name = name

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        effective_epoch = epoch or _J2000

        mass_kg = self._drag_config.mass_kg if self._drag_config else _DEFAULT_SAT_MASS_KG
        radius_m = _DEFAULT_SAT_RADIUS_M
        if self._drag_config:
            radius_m = math.sqrt(self._drag_config.area_m2 / math.pi)

        sim = _build_simulation(satellites, effective_epoch, mass_kg, radius_m, self._name)
        sim_json = json.dumps(sim, indent=2, ensure_ascii=False)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("simulation.json", sim_json)
            zf.writestr("version.ini", _VERSION_INI)
            zf.writestr("info.json", _build_info_json(self._name))
            zf.writestr("ui-state.json", _UI_STATE_JSON)

        return len(satellites)


_VERSION_INI = "Universe Sandbox \u00b2\nUniverse Sandbox Update 35.4.5\n45832\n"


def _build_info_json(name: str) -> str:
    return json.dumps({
        "Name": name,
        "Description": "",
        "Metadata": "",
        "IconPath": "",
        "ContentPath": "",
        "ChangeNote": "",
        "PublishedFileId": "0",
        "CreatorSteamId": "0",
        "PackageURL": "",
        "ImageURL": "",
        "InstalledSizeOnDisk": "0",
        "InstalledTimestamp": "/Date(-62135568000000-0800)/",
        "IsOwned": "True",
        "Flags": "None",
        "Visibility": "k_ERemoteStoragePublishedFileVisibilityPublic",
        "Tags": [{"$v": "Simulations"}],
        "TimeUpdated": "0",
    }, indent=2)


_UI_STATE_JSON = json.dumps({
    "Selector": {"0": _EARTH_ID},
    "DockPanelLeft": {
        "views": [],
        "containerSize": 0,
        "scrollPos": 0,
        "minimized": False,
    },
    "Windows": {},
    "ActiveTool": {"$type": 1},
}, indent=2)
