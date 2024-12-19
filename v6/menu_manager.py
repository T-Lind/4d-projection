import pygame
from typing import List
from asset_manager import AssetManager
from level_manager import LevelManager
import os

class MenuManager:
    def __init__(self, level_manager: LevelManager, assets: AssetManager, in_game=False):
        self.level_manager = level_manager
        self.assets = assets
        self.in_game = in_game
        self.screen = pygame.display.get_surface()
        self.font = self.assets.get_font('pixel_24')
        self.title_font = self.assets.get_font('pixel_64')
        if self.in_game:
            self.options = ['Resume', 'Return to Main Menu']
        else:
            self.options = self._load_levels()
        self.selected_index = 0
        self.running = True
        self.resume_game = False
        self.return_to_main = False


    def run_pause_menu(self, screen):
        clock = pygame.time.Clock()
        while self.running:
            self._handle_pause_events()
            self._render_pause_menu()
            pygame.display.flip()
            clock.tick(60)

    def _handle_pause_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.resume_game = True
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.selected_index = (self.selected_index - 1) % len(self.options)
                    self.assets.play_sound('highlight')
                elif event.key == pygame.K_DOWN:
                    self.selected_index = (self.selected_index + 1) % len(self.options)
                    self.assets.play_sound('highlight')
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.assets.play_sound('select')
                    if self.selected_index == 0:  # Resume
                        self.resume_game = True
                    elif self.selected_index == 1:  # Return to Main Menu
                        self.return_to_main = True
                    self.running = False

    def _load_levels(self) -> List[str]:
        levels = []
        level_folder = self.level_manager.level_folder
        for filename in os.listdir(level_folder):
            if filename.endswith('.json'):
                levels.append(filename.replace('.json', ''))
        levels.sort(key=lambda x: int(x))
        return levels

    def run(self) -> int:
        clock = pygame.time.Clock()
        while self.running:
            self._handle_events()
            self._render()
            pygame.display.flip()
            clock.tick(60)
        return int(self.options[self.selected_index])

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_index = (self.selected_index - 1) % len(self.options)
                    self.assets.play_sound('highlight')
                elif event.key == pygame.K_DOWN:
                    self.selected_index = (self.selected_index + 1) % len(self.options)
                    self.assets.play_sound('highlight')
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    self.assets.play_sound('select')
                    self.running = False

    def _render_pause_menu(self):
        self.screen.fill((0, 0, 0))
        title_surface = self.title_font.render("Paused", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(self.screen.get_width() // 2, 150))
        self.screen.blit(title_surface, title_rect)

        for idx, option in enumerate(self.options):
            if idx == self.selected_index:
                color = (255, 215, 0)  # Highlighted color
            else:
                color = (255, 255, 255)
            option_surface = self.font.render(option, True, color)
            option_rect = option_surface.get_rect(center=(self.screen.get_width() // 2, 300 + idx * 50))
            self.screen.blit(option_surface, option_rect)


    def _render(self):
        self.screen.fill((0, 0, 0))
        title_surface = self.title_font.render("Rotander", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(self.screen.get_width() // 2, 100))
        self.screen.blit(title_surface, title_rect)

        for idx, option in enumerate(self.options):
            if idx == self.selected_index:
                color = (255, 215, 0)  # Highlighted color
            else:
                color = (255, 255, 255)
            option_surface = self.font.render(f"Level {option}", True, color)
            option_rect = option_surface.get_rect(center=(self.screen.get_width() // 2, 200 + idx * 40))
            self.screen.blit(option_surface, option_rect)