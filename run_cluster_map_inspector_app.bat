@echo off
rem Launches cluster_map_inspector_app.py on port 5006 and opens it in the browser.
rem Run from an Anaconda/Miniforge Prompt with the twist-segmenter environment active.
cd /d "%~dp0"
panel serve cluster_map_inspector_app.py --port 5006 --show
