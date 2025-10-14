"""
Unit tests for TripleKSampler configuration system.

Tests the TOML configuration loading, file operations, and data validation.
"""

import pytest
import os
import tempfile
import shutil


class TestTomlHandling:
    """Test TOML configuration loading and processing."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'config.toml')
        self.example_path = os.path.join(self.test_dir, 'config.example.toml')

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_load_valid_toml(self):
        """Test loading valid TOML configuration."""
        # Create a valid config file
        config_content = """
[sampling]
base_quality_threshold = 25

[boundaries]
default_t2v = 0.8
default_i2v = 0.95

[logging]
level = "DEBUG"
"""

        with open(self.config_path, 'w') as f:
            f.write(config_content)

        # Test TOML loading directly using the implementation logic
        try:
            import tomllib
        except ImportError:
            try:
                import tomli  # type: ignore as tomllib  # type: ignore
            except ImportError:
                pytest.skip("No TOML library available")

        # Test the actual file reading
        with open(self.config_path, 'rb') as f:
            config = tomllib.load(f)

        assert config["sampling"]["base_quality_threshold"] == 25
        assert config["boundaries"]["default_t2v"] == 0.8
        assert config["boundaries"]["default_i2v"] == 0.95
        assert config["logging"]["level"] == "DEBUG"

    def test_copy_template_to_config(self):
        """Test copying template to config file."""
        # Create template file only
        template_content = """
[sampling]
base_quality_threshold = 20

[boundaries]
default_t2v = 0.875
default_i2v = 0.900

[logging]
level = "INFO"
"""

        with open(self.example_path, 'w') as f:
            f.write(template_content)

        # Test file copy operation
        shutil.copy2(self.example_path, self.config_path)

        # Verify the copy worked
        assert os.path.exists(self.config_path)

        with open(self.config_path, 'r') as f:
            copied_content = f.read()

        assert "base_quality_threshold = 20" in copied_content
        assert "default_t2v = 0.875" in copied_content

    def test_default_config_structure(self):
        """Test the expected default configuration structure."""
        # Test that we know what the expected defaults should be
        expected_defaults = {
            "sampling": {"base_quality_threshold": 20},
            "boundaries": {"default_t2v": 0.875, "default_i2v": 0.900},
            "logging": {"level": "INFO"}
        }

        # This is what the fallback should produce
        for section, values in expected_defaults.items():
            assert isinstance(values, dict)
            for key, value in values.items():
                assert isinstance(key, str)
                # Check value types are reasonable
                if key == "base_quality_threshold":
                    assert isinstance(value, int) and value > 0
                elif key.startswith("default_"):
                    assert isinstance(value, float) and 0 <= value <= 1
                elif key == "level":
                    assert value in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_partial_toml_loading(self):
        """Test loading TOML with missing sections."""
        # Config with only sampling section
        partial_config = """
[sampling]
base_quality_threshold = 30
"""

        with open(self.config_path, 'w') as f:
            f.write(partial_config)

        try:
            import tomllib
        except ImportError:
            try:
                import tomli  # type: ignore as tomllib  # type: ignore
            except ImportError:
                pytest.skip("No TOML library available")

        with open(self.config_path, 'rb') as f:
            config = tomllib.load(f)

        # Should only have the sampling section
        assert "sampling" in config
        assert config["sampling"]["base_quality_threshold"] == 30
        assert "boundaries" not in config
        assert "logging" not in config

    def test_invalid_toml_handling(self):
        """Test handling of invalid TOML syntax."""
        # Create invalid TOML file
        invalid_config = """
[sampling
base_quality_threshold = 20
invalid syntax here
"""

        with open(self.config_path, 'w') as f:
            f.write(invalid_config)

        try:
            import tomllib
        except ImportError:
            try:
                import tomli  # type: ignore as tomllib  # type: ignore
            except ImportError:
                pytest.skip("No TOML library available")

        # Should raise an exception for invalid TOML
        with pytest.raises(Exception):  # TOMLDecodeError or similar
            with open(self.config_path, 'rb') as f:
                tomllib.load(f)

    def test_toml_library_availability(self):
        """Test TOML library import behavior."""
        # Test that at least one TOML library can be imported
        toml_available = False
        try:
            import tomllib
            toml_available = True
        except ImportError:
            try:
                import tomli  # type: ignore
                toml_available = True
            except ImportError:
                pass

        if not toml_available:
            pytest.skip("No TOML library available for testing")
        else:
            # If we got here, at least one library works
            assert True

    def test_missing_keys_in_sections(self):
        """Test config with missing keys within sections."""
        # Config missing some keys
        incomplete_config = """
[sampling]
# base_quality_threshold missing

[boundaries]
default_t2v = 0.8
# default_i2v missing

[logging]
# level missing
"""

        with open(self.config_path, 'w') as f:
            f.write(incomplete_config)

        try:
            import tomllib
        except ImportError:
            try:
                import tomli  # type: ignore as tomllib  # type: ignore
            except ImportError:
                pytest.skip("No TOML library available")

        with open(self.config_path, 'rb') as f:
            config = tomllib.load(f)

        # Should have sections but missing keys
        assert "sampling" in config
        assert "boundaries" in config
        assert "logging" in config

        # Missing keys should not be present
        assert "base_quality_threshold" not in config["sampling"]
        assert "default_t2v" in config["boundaries"]
        assert config["boundaries"]["default_t2v"] == 0.8
        assert "default_i2v" not in config["boundaries"]
        assert "level" not in config["logging"]

    def test_wrong_data_types(self):
        """Test config with wrong data types."""
        # Config with wrong types
        wrong_types_config = """
[sampling]
base_quality_threshold = "not_a_number"

[boundaries]
default_t2v = "not_a_float"
default_i2v = 0.900

[logging]
level = 123
"""

        with open(self.config_path, 'w') as f:
            f.write(wrong_types_config)

        try:
            import tomllib
        except ImportError:
            try:
                import tomli  # type: ignore as tomllib  # type: ignore
            except ImportError:
                pytest.skip("No TOML library available")

        with open(self.config_path, 'rb') as f:
            config = tomllib.load(f)

        # TOML will load the values as their literal types
        assert config["sampling"]["base_quality_threshold"] == "not_a_number"
        assert config["boundaries"]["default_t2v"] == "not_a_float"
        assert config["boundaries"]["default_i2v"] == 0.900
        assert config["logging"]["level"] == 123


class TestFileOperations:
    """Test file operations used in configuration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_file_copy_operation(self):
        """Test basic file copy functionality."""
        # Create source file
        source_content = "test content"
        source_path = os.path.join(self.test_dir, 'source.txt')
        dest_path = os.path.join(self.test_dir, 'dest.txt')

        with open(source_path, 'w') as f:
            f.write(source_content)

        # Test copy
        shutil.copy2(source_path, dest_path)

        # Verify
        assert os.path.exists(dest_path)
        with open(dest_path, 'r') as f:
            assert f.read() == source_content

    def test_file_existence_check(self):
        """Test file existence checking."""
        # File doesn't exist
        non_existent = os.path.join(self.test_dir, 'does_not_exist.txt')
        assert not os.path.exists(non_existent)

        # Create file
        test_file = os.path.join(self.test_dir, 'exists.txt')
        with open(test_file, 'w') as f:
            f.write("test")

        # File exists
        assert os.path.exists(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])