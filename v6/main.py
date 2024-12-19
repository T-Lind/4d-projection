import sys
import pygame
from level_manager import LevelManager
from settings import Settings
from viewer import PlaneSliceViewer
from menu_manager import MenuManager
from asset_manager import AssetManager

def main():
    pygame.init()
    pygame.display.set_caption("Rotander")

    assets = AssetManager()
    level_manager = LevelManager()
    menu = MenuManager(level_manager, assets)
    running = True

    while running:
        if level_manager.current_level is None:
            start_level = menu.run()
            level_manager.current_level = start_level
        level_path = level_manager.get_current_level_path()
        try:
            settings = Settings(config_path=level_path)
            viewer = PlaneSliceViewer(settings, level_manager, assets)
            viewer.run()
            if viewer.level_complete:
                if level_manager.has_next_level():
                    level_manager.advance_level()
                else:
                    # Game completed, return to main menu
                    level_manager.current_level = None
            else:
                if viewer.running:
                    # Player returned to main menu from pause menu
                    level_manager.current_level = None
                else:
                    # Player quit the game
                    running = False
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()