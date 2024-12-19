import sys
import pygame
from level_manager import LevelManager
from settings import Settings
from viewer import PlaneSliceViewer
from menu_manager import MenuManager
from asset_manager import AssetManager

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Rotander")

    assets = AssetManager()
    assets.play_music()
    level_manager = LevelManager()
    running = True

    while running:
        if level_manager.current_level is None:
            menu = MenuManager(level_manager, assets)
            selected_level = menu.run()
            level_manager.current_level = selected_level

        level_path = level_manager.get_current_level_path()
        try:
            settings = Settings(config_path=level_path)
            viewer = PlaneSliceViewer(settings, level_manager, assets)
            viewer.run()
            if viewer.return_to_main_menu:
                level_manager.current_level = None
            elif viewer.level_complete:
                if level_manager.has_next_level():
                    level_manager.advance_level()
                else:
                    level_manager.current_level = None  # Return to main menu
            elif not viewer.running:
                running = False  # Player quit the game
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()