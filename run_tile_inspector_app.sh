#!/bin/bash
# Launches tile_inspector_app.py on its semi-permanently assigned port, 5009.
cd "$(dirname "$0")"
panel serve tile_inspector_app.py --port 5009 --show
