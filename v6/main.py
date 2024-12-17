import argparse
import sys
from settings import Settings
from viewer import PlaneSliceViewer

def main():
    parser = argparse.ArgumentParser(description="Vertical Plane Slice Viewer with Enhanced Features")
    parser.add_argument("file", nargs='?', default='v6/shapes_config.json', help="Path to config JSON file")
    args = parser.parse_args()

    try:
        settings = Settings(args.file)
        viewer = PlaneSliceViewer(settings)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()