"""Pygame viewer for RobotaxiEnv; actions are [acceleration, steering]."""

import math

import jax
import jax.numpy as jnp
import pygame


WHITE = (245, 247, 250)
BLACK = (25, 30, 40)
GRAY = (75, 85, 100)
RED = (215, 70, 65)
BLUE = (45, 105, 210)
GREEN = (55, 170, 95)
YELLOW = (255, 210, 65)
PATH_COLOR = (115, 170, 235)
PANEL_COLOR = (25, 30, 40)


class PygameFrontend:
    """agent_fn, if supplied, receives observation.as_vector()."""

    def __init__(self, env, env_params, init_state,
                 eval_mode=False, agent_fn=None):
        if eval_mode and agent_fn is None:
            raise ValueError("eval_mode=True requires agent_fn")

        pygame.init()
        self.env = env
        self.params = env_params
        self.init_state = init_state
        self.eval_mode = eval_mode
        self.agent_fn = agent_fn
        self.height, self.width = map(int, env_params.map_height_width)
        self.CELL_SIZE = 40
        self.map_height_px = self.height * self.CELL_SIZE
        self.panel_height = 140
        self.screen = pygame.display.set_mode((
            self.width * self.CELL_SIZE,
            self.map_height_px + self.panel_height,
        ))
        pygame.display.set_caption("Robotaxi - bicycle model")
        self.font = pygame.font.Font(None, 22)
        self.clock = pygame.time.Clock()
        self.ray_surface = pygame.Surface(
            (self.screen.get_width(), self.map_height_px), pygame.SRCALPHA
        )
        # Keep the original viewer's display scale; physics belongs to env.
        self.radius_px = self.CELL_SIZE // 2
        self.key = jax.random.PRNGKey(0)
        self.running = True
        self.paused = False
        self.show_rays = True
        self._held_scancodes = set()
        self._down_keys = set()
        self.status_text = ""
        self.status_until = 0
        self.episodes = 0
        self.last_result = "-"
        self.last_return = 0.0
        self.reset()

    def next_key(self):
        self.key, key = jax.random.split(self.key)
        return key

    def reset(self):
        self.obs, self.state = self.env.reset(
            self.next_key(), self.params, self.init_state
        )
        self.info = {}
        self.reward = 0.0
        self.episode_return = 0.0
        self.status_text = "Reset to start"
        self.status_until = pygame.time.get_ticks() + 1500

    def to_pixel(self, position):
        x, y = float(position[0]), float(position[1])
        return (
            round((x + 0.5) * self.CELL_SIZE),
            round((self.height - 0.5 - y) * self.CELL_SIZE),
        )

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        accelerate = bool(
            keys[pygame.K_w] or keys[pygame.K_UP]
            or pygame.KSCAN_W in self._held_scancodes
        )
        brake = bool(
            keys[pygame.K_s] or keys[pygame.K_DOWN]
            or pygame.KSCAN_S in self._held_scancodes
        )
        left = bool(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
            or pygame.KSCAN_A in self._held_scancodes
        )
        right = bool(
            keys[pygame.K_d] or keys[pygame.K_RIGHT]
            or pygame.KSCAN_D in self._held_scancodes
        )
        # Independent channels: full throttle and full steering can coexist.
        return jnp.array([
            float(accelerate) - float(brake),
            float(left) - float(right),
        ], dtype=jnp.float32)

    @staticmethod
    def _matches(event, key, scancode, letters):
        """Accept logical keys, physical keys, and translated text."""
        return (
            event.key == key
            or getattr(event, "scancode", None) == scancode
            or getattr(event, "unicode", "").casefold() in letters
        )

    def handle_event(self, event):
        """Return True when a reset should suppress this frame's step."""
        if event.type == pygame.QUIT:
            self.running = False
            return False
        if event.type == pygame.WINDOWFOCUSLOST:
            self._held_scancodes.clear()
            self._down_keys.clear()
            self.paused = True
            return False
        if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
            return False

        scancode = getattr(event, "scancode", None)
        token = ("scan", scancode) if scancode else ("key", event.key)
        if event.type == pygame.KEYUP:
            self._down_keys.discard(token)
            self._held_scancodes.discard(scancode)
            return False

        repeated = token in self._down_keys or getattr(event, "repeat", False)
        self._down_keys.add(token)
        if scancode:
            self._held_scancodes.add(scancode)
        if repeated:
            return False
        if getattr(event, "mod", 0) & (pygame.KMOD_CTRL | pygame.KMOD_META):
            return False

        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif (
            event.key == pygame.K_F2
            or self._matches(event, pygame.K_r, pygame.KSCAN_R, {"r", "к"})
        ):
            self.reset()
            return True
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif (
            event.key == pygame.K_F3
            or self._matches(event, pygame.K_l, pygame.KSCAN_L, {"l", "д"})
        ):
            self.show_rays = not self.show_rays
        return False

    def advance(self, action):
        self.obs, self.state, reward, done, self.info = self.env.step(
            self.next_key(), self.state, action, self.params
        )
        self.reward = float(reward)
        self.episode_return += self.reward

        if bool(done):
            self.episodes += 1
            self.last_return = self.episode_return
            if bool(self.info.get("success", False)):
                self.last_result = "Goal reached"
            elif bool(self.info.get("collision", False)):
                self.last_result = "Collision"
            else:
                self.last_result = "Time limit"
            self.episode_return = 0.0
            self.obs, self.state = self.env.reset(
                self.next_key(), self.params, self.init_state
            )

    def draw_rays(self, state, observation):
        self.ray_surface.fill((0, 0, 0, 0))
        start = self.to_pixel(state.agent_pos)
        for ray in observation.collision_rays:
            end = self.to_pixel(state.agent_pos + ray)
            pygame.draw.line(
                self.ray_surface, (215, 70, 65, 85), start, end, 1
            )
        self.screen.blit(self.ray_surface, (0, 0))

    def draw_info(self, state, observation):
        speed = float(state.speed)
        steering = math.degrees(float(state.steering_angle))
        dx, dy = state.goal_pos - state.agent_pos
        distance = math.hypot(float(dx), float(dy))
        mode = "Agent" if self.eval_mode else "Manual"
        if self.paused:
            mode += " / PAUSED"
        if pygame.time.get_ticks() < self.status_until:
            mode += f" / {self.status_text}"
        rays_status = "ON" if self.show_rays else "OFF"

        lines = [
            f"Speed: {speed:+.2f} | Steering: {steering:+.1f} deg | "
            f"Reward: {self.reward:+.3f} | Return: {self.episode_return:+.2f}",
            f"Step: {int(state.time)} | Goal distance: {distance:.2f} | "
            f"Route distance: {float(observation.distance_to_path):.2f}",
            f"{mode} | Episodes: {self.episodes} | "
            f"Last: {self.last_result} ({self.last_return:+.2f})",
            "W / Up: accelerate | S / Down: brake / reverse | A / D: steer",
            f"R / F2: reset | Space: pause | L / F3: rays ({rays_status}) | Esc: exit",
        ]
        pygame.draw.rect(self.screen, PANEL_COLOR, (
            0, self.map_height_px, self.screen.get_width(), self.panel_height
        ))
        for index, text in enumerate(lines):
            image = self.font.render(text, True, WHITE)
            self.screen.blit(image, (12, self.map_height_px + 12 + index * 24))

    def draw(self):
        # Transfer arrays to the CPU once per frame, outside the physics code.
        state, observation = jax.device_get((self.state, self.obs))
        self.screen.fill(WHITE)
        points = [self.to_pixel(state.agent_pos)]
        points.extend(self.to_pixel(point) for point in state.path_array)
        if len(points) > 1:
            pygame.draw.lines(self.screen, PATH_COLOR, False, points, 3)

        for positions, color in (
            (state.static_obstacles, GRAY),
            (state.kinematic_obstacles, RED),
        ):
            for position in positions:
                center = self.to_pixel(position)
                pygame.draw.circle(self.screen, color, center, self.radius_px)
                pygame.draw.circle(self.screen, BLACK, center, self.radius_px, 1)

        pygame.draw.circle(
            self.screen, GREEN, self.to_pixel(state.goal_pos), self.radius_px
        )
        if self.show_rays:
            self.draw_rays(state, observation)

        center = self.to_pixel(state.agent_pos)
        pygame.draw.circle(self.screen, BLUE, center, self.radius_px)
        pygame.draw.circle(self.screen, BLACK, center, self.radius_px, 1)
        forward = state.agent_forward_dir
        end = (
            center[0] + round(float(forward[0]) * self.radius_px),
            center[1] - round(float(forward[1]) * self.radius_px),
        )
        pygame.draw.line(self.screen, YELLOW, center, end, 3)
        self.draw_info(state, observation)
        pygame.display.flip()

    def run(self):
        try:
            while self.running:
                reset_this_frame = False
                for event in pygame.event.get():
                    reset_this_frame = self.handle_event(event) or reset_this_frame

                if not self.running:
                    break
                if not self.paused and not reset_this_frame:
                    if self.eval_mode:
                        action = self.agent_fn(self.obs.as_vector())
                    else:
                        action = self.handle_keys()
                    self.advance(action)

                self.draw()
                self.clock.tick(int(self.params.fps))
        finally:
            pygame.quit()
