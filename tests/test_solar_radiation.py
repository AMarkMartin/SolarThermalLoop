"""
Unit tests for solar radiation model
Validates sun position calculations and irradiance components
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../')

from src.models import SolarRadiationModel, LocationParams


class TestSolarPosition:
    """Test sun position calculations against known values"""

    def test_solar_declination_summer_solstice(self):
        """Test declination at summer solstice (day 172)"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # Summer solstice: ~June 21 (day 172)
        delta = model.solar_declination(172)

        # Should be close to +23.45 degrees
        assert abs(delta - 23.45) < 1.0, f"Summer solstice declination {delta}° should be ~23.45°"

    def test_solar_declination_winter_solstice(self):
        """Test declination at winter solstice (day 355)"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # Winter solstice: ~Dec 21 (day 355)
        delta = model.solar_declination(355)

        # Should be close to -23.45 degrees
        assert abs(delta + 23.45) < 1.0, f"Winter solstice declination {delta}° should be ~-23.45°"

    def test_solar_declination_equinox(self):
        """Test declination at equinox"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # Spring equinox: ~March 20 (day 79)
        delta = model.solar_declination(79)

        # Should be close to 0 degrees
        assert abs(delta) < 2.0, f"Equinox declination {delta}° should be ~0°"

    def test_solar_altitude_solar_noon(self):
        """Test altitude at solar noon"""
        # 40°N latitude
        location = LocationParams(latitude=40.0)
        model = SolarRadiationModel(location)

        # Summer solstice, solar noon
        altitude, azimuth = model.solar_position(12.0, 172)

        # At solar noon on summer solstice:
        # altitude = 90° - latitude + declination
        # = 90 - 40 + 23.45 = 73.45°
        expected_altitude = 90 - 40 + 23.45
        assert abs(altitude - expected_altitude) < 3.0, \
            f"Solar noon altitude {altitude}° should be ~{expected_altitude}°"

    def test_solar_altitude_negative_at_night(self):
        """Sun should be below horizon at night"""
        location = LocationParams(latitude=40.0)
        model = SolarRadiationModel(location)

        # Midnight on any day
        altitude, azimuth = model.solar_position(0.0, 180)

        assert altitude < 0, f"Altitude at midnight {altitude}° should be negative"

    def test_solar_position_symmetry(self):
        """Test morning/afternoon symmetry"""
        location = LocationParams(latitude=40.0)
        model = SolarRadiationModel(location)

        # 2 hours before and after solar noon
        alt_morning, az_morning = model.solar_position(10.0, 180)
        alt_afternoon, az_afternoon = model.solar_position(14.0, 180)

        # Altitudes should be nearly equal (equation of time causes slight asymmetry)
        assert abs(alt_morning - alt_afternoon) < 2.0, \
            "Morning and afternoon altitudes should be symmetric"


class TestClearSkyRadiation:
    """Test clear sky irradiance calculations"""

    def test_zero_irradiance_at_night(self):
        """Irradiance should be zero when sun is below horizon"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # Midnight
        irrad = model.clear_sky_radiation(altitude=-20.0)

        assert irrad == 0.0, "Irradiance should be zero at night"

    def test_maximum_irradiance_at_high_altitude(self):
        """Maximum irradiance at high sun angles"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # 90° altitude (sun directly overhead)
        irrad = model.clear_sky_radiation(altitude=90.0)

        # Should be close to solar constant * atmospheric transmission
        # At Denver elevation (1600m), atmospheric path is shorter but
        # the model's tau factor can reduce DNI below sea-level values
        assert 750 < irrad < 1200, f"Clear sky DNI {irrad} W/m² should be 750-1200"

    def test_lower_irradiance_at_low_altitude(self):
        """Irradiance decreases at low sun angles (longer path)"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        irrad_high = model.clear_sky_radiation(altitude=60.0)
        irrad_low = model.clear_sky_radiation(altitude=15.0)

        assert irrad_high > irrad_low, \
            "Irradiance should be higher at higher sun angles"


class TestTiltedSurfaceIrradiance:
    """Test irradiance on tilted surfaces"""

    def test_cloud_cover_reduces_direct(self):
        """Cloud cover should reduce direct component"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # Clear sky
        irrad_clear = model.calculate_irradiance(
            time_hours=12.0,
            day_of_year=180,
            cloud_cover=0.0
        )

        # Overcast
        irrad_cloudy = model.calculate_irradiance(
            time_hours=12.0,
            day_of_year=180,
            cloud_cover=0.9
        )

        assert irrad_clear['direct'] > irrad_cloudy['direct'], \
            "Direct radiation should decrease with clouds"

    def test_diffuse_increases_with_clouds(self):
        """Diffuse component increases with cloud cover"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        # Clear sky
        irrad_clear = model.calculate_irradiance(
            time_hours=12.0,
            day_of_year=180,
            cloud_cover=0.0
        )

        # Partly cloudy
        irrad_cloudy = model.calculate_irradiance(
            time_hours=12.0,
            day_of_year=180,
            cloud_cover=0.5
        )

        assert irrad_cloudy['diffuse'] > irrad_clear['diffuse'], \
            "Diffuse radiation should increase with clouds"

    def test_zero_at_night(self):
        """All components should be zero at night"""
        location = LocationParams()
        model = SolarRadiationModel(location)

        irrad = model.calculate_irradiance(
            time_hours=0.0,  # Midnight
            day_of_year=180
        )

        assert irrad['total'] == 0.0, "Total irradiance should be zero at night"
        assert irrad['direct'] == 0.0, "Direct should be zero at night"
        assert irrad['diffuse'] == 0.0, "Diffuse should be zero at night"

    def test_tilted_surface_higher_in_winter(self):
        """Tilted surface should collect more in winter (low sun)"""
        location = LocationParams(latitude=40.0)
        model = SolarRadiationModel(location)

        # Winter day, solar noon
        # Horizontal surface
        irrad_horiz = model.calculate_irradiance(
            time_hours=12.0,
            day_of_year=355,
            panel_tilt=0.0
        )

        # Tilted at latitude
        irrad_tilt = model.calculate_irradiance(
            time_hours=12.0,
            day_of_year=355,
            panel_tilt=40.0
        )

        assert irrad_tilt['direct'] > irrad_horiz['direct'], \
            "Tilted surface should collect more direct radiation in winter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
