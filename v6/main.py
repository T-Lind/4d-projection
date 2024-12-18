import argparse
import sys
from level_manager import LevelManager
from settings import Settings
from viewer import PlaneSliceViewer

def main():
    parser = argparse.ArgumentParser(description="Vertical Plane Slice Viewer with Enhanced Features")
    parser.add_argument("--start-level", type=int, default=1, help="Level number to start from")
    args = parser.parse_args()
    
    level_manager = LevelManager(start_level=args.start_level)
    
    while level_manager.has_next_level():
        level_path = level_manager.get_current_level_path()
        try:
            settings = Settings(config_path=level_path)
            viewer = PlaneSliceViewer(settings, level_manager)
            viewer.run()
            if viewer.level_complete:
                level_manager.advance_level()
            else:
                break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    print("All levels completed!")

if __name__ == "__main__":
    main()