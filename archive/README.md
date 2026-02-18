# Solar Thermal Simulation - Version Archive

This directory contains archived versions of the solar thermal simulation project.

## Version History

### v1.0_baseline (2026-02-14)
**Baseline implementation meeting minimum requirements**
- Complete solar panel → storage tank heat transfer simulation
- Thermodynamically accurate energy balances
- 24-hour simulation with visualization
- System efficiency: 37.5%
- See: [v1.0_baseline/VERSION_NOTES.md](v1.0_baseline/VERSION_NOTES.md)

## Archive Structure

Each version folder contains:
- All source code files from that version
- Configuration files (pyproject.toml, requirements.txt)
- Sample output/results
- VERSION_NOTES.md documenting features and test results

## Restoring a Version

To restore any archived version:
```bash
cp archive/<version>/* .
```

## Current Development

The main project directory contains the latest development version.
Always check this archive before making significant changes to preserve working versions.
