#!/bin/bash
# Launches cluster_map_inspector_app.py (the pixel-resolution phase/cluster
# map app) on its semi-permanently assigned port, 5006.
cd "$(dirname "$0")"
panel serve cluster_map_inspector_app.py --port 5006 --show
