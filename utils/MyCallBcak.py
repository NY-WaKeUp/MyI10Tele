import glfw
import mujoco
from threading import Lock

class MyCallbacks:
    def __init__(self, hide_menus):
        self._gui_lock                   = Lock()
        self._button_left_pressed        = False
        self._button_right_pressed       = False
        self._left_double_click_pressed  = False
        self._right_double_click_pressed = False
        self._last_left_click_time       = None
        self._last_right_click_time      = None
        self._last_mouse_x               = 0
        self._last_mouse_y               = 0
        self._paused                     = False
        self._render_every_frame         = True
        self._time_per_render            = 1/60.0
        self._run_speed                  = 1.0
        self._loop_count                 = 0
        self._advance_by_one_step        = False
        # Keyboard 
        self._key_pressed                = None
        self._is_key_pressed             = False
        # Keyboard buffer
        self._key_pressed_set            = set()
        self._key_repeated_set           = set()
        
    def _key_callback(self, window, key, scancode, action, mods):
        """
            Key callback        
        """

        # Flags for key pressed 
        is_key_pressed  = (action==glfw.PRESS)
        is_key_released = (action==glfw.RELEASE)
        is_key_repeated = (action==glfw.REPEAT)
        
        # Add and discard keys
        if is_key_pressed:
            self._key_pressed_set.add(key)
        if is_key_repeated:
            self._key_repeated_set.add(key)
        if is_key_released:
            # Remove from pressed and repeated lists (if present)
            self._key_pressed_set.discard(key)
            self._key_repeated_set.discard(key)
        
        # Pause / resume handling (space)
        # if is_key_pressed and (key==glfw.KEY_SPACE) and (self._paused is not None):
        #     self._paused = not self._paused

        # Quit (escape)
        if (key==glfw.KEY_ESCAPE):
            glfw.set_window_should_close(self.window, True)

        # Store key pressed (legacy)
        self._key_pressed    = key 
        self._is_key_pressed = True
        
        # Return
        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = (time_now - self._last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = (time_now - self._last_right_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb
        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)
