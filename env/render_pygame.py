import pygame

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

CELL_SIZE = 32  # tamaño de cada celda en píxeles

# Colores fallback (si no usas imágenes)
COLORS = {
    "OPEN": (200, 200, 200),
    "BUSH": (34, 139, 34),
    "FOREST": (0, 100, 0),
    "RUBBLE": (139, 69, 19),
    "WALL": (100, 100, 100),
    "WATER": (0, 0, 255)
}


# ==========================================================
# CLASE RENDERER
# ==========================================================

class Renderer:
    def __init__(self, grid, points, blue_team=None, red_team=None):
        """
        grid: mapa (lista 2D)
        points: diccionario con A, B, C
        blue_team: lista de pelotones azules [{"pos": (x,y)}]
        red_team: lista de enemigos [{"pos": (x,y)}]
        """

        pygame.init()

        self.grid = grid
        self.points = points
        self.blue_team = blue_team if blue_team else []
        self.red_team = red_team if red_team else []

        self.size = len(grid)

        # Crear ventana
        self.screen = pygame.display.set_mode(
            (self.size * CELL_SIZE, self.size * CELL_SIZE)
        )
        pygame.display.set_caption("Normandy Render")

        # Cargar imágenes
        self._load_images()

    # ==========================================================
    # CARGA DE IMÁGENES
    # ==========================================================
    def _load_images(self):
        """
        Carga imágenes del proyecto (si existen)
        """

        try:
            self.tiger = pygame.image.load("resources/tiger.png")
            self.tiger = pygame.transform.scale(self.tiger, (CELL_SIZE, CELL_SIZE))
        except:
            self.tiger = None

        try:
            self.sherman = pygame.image.load("resources/sherman.png")
            self.sherman = pygame.transform.scale(self.sherman, (CELL_SIZE, CELL_SIZE))
        except:
            self.sherman = None

    # ==========================================================
    # DIBUJAR MAPA
    # ==========================================================
    def draw_map(self):
        """
        Dibuja el grid del mapa
        """

        for y in range(self.size):
            for x in range(self.size):

                cell = self.grid[y][x]
                color = COLORS[cell["type"]]

                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

                # borde de celda
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1
                )

    # ==========================================================
    # DIBUJAR PUNTOS (A, B, C)
    # ==========================================================
    def draw_points(self):
        """
        Dibuja los puntos de captura
        """

        for name, (x, y) in self.points.items():

            # B más importante → rojo
            color = (255, 0, 0) if name == "B" else (255, 165, 0)

            pygame.draw.circle(
                self.screen,
                color,
                (x * CELL_SIZE + CELL_SIZE // 2,
                 y * CELL_SIZE + CELL_SIZE // 2),
                8
            )

    # ==========================================================
    # DIBUJAR PELOTONES AZULES
    # ==========================================================
    def draw_blue_team(self):
        """
        Dibuja los pelotones azules
        """

        for peloton in self.blue_team:
            x, y = peloton["pos"]

            if self.tiger:
                self.screen.blit(self.tiger, (x * CELL_SIZE, y * CELL_SIZE))
            else:
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 255),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

    # ==========================================================
    # DIBUJAR ENEMIGOS ROJOS
    # ==========================================================
    def draw_red_team(self):
        """
        Dibuja los enemigos
        """

        for enemy in self.red_team:
            x, y = enemy["pos"]

            if self.sherman:
                self.screen.blit(self.sherman, (x * CELL_SIZE, y * CELL_SIZE))
            else:
                pygame.draw.rect(
                    self.screen,
                    (255, 0, 0),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

    # ==========================================================
    # RENDER COMPLETO
    # ==========================================================
    def render(self):
        """
        Dibuja todo en pantalla
        """

        self.screen.fill((0, 0, 0))

        self.draw_map()
        self.draw_points()
        self.draw_blue_team()
        self.draw_red_team()

        pygame.display.flip()

    # ==========================================================
    # LOOP PRINCIPAL
    # ==========================================================
    def run(self):
        """
        Loop principal de pygame
        """

        clock = pygame.time.Clock()
        running = True

        while running:
            clock.tick(30)  # FPS

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.render()

        pygame.quit()