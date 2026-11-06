@echo off
rem Launches tile_inspector_app.py on port 5009 and opens it in the browser.
rem Run from an Anaconda/Miniforge Prompt with the twist-segmenter environment active.
cd /d "%~dp0"
panel serve tile_inspector_app.py --port 5009 --show
