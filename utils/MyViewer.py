from utils.MyCallBcak import MyCallbacks
import glfw
import mujoco
import pathlib
import cv2
import numpy as np
import time

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split(".")))


class MyViewer(MyCallbacks):
    def __init__(
        self,
        model,
        data,
        mode="window",
        title="My Viewer",
        width=None,
        height=None,
        hide_menus=True,
        maxgeom=10000,
        n_fig=1,
        perturbation=True,
        use_rgb_overlay=True,
        loc_rgb_overlay="top right",
    ):
        super().__init__(hide_menus)

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in ["window"]:
            raise NotImplementedError("Invalid mode. Only 'window' is supported.")

        # keep true while running
        self.is_alive = True

        self.CONFIG_PATH = pathlib.Path.joinpath(
            pathlib.Path.home(), ".config/mujoco_viewer/config.yaml"
        )

        # glfw init
        glfw.init()

        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if self.render_mode == "offscreen":
            glfw.window_hint(glfw.VISIBLE, 0)

        # Create window
        self.maxgeom = maxgeom
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        # install callbacks only for 'window' mode
        if self.render_mode == "window":
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width * 1.0 / window_width

            # set callbacks
            glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=self.maxgeom)
        self.pert = mujoco.MjvPerturb()

        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )

        width, height = glfw.get_framebuffer_size(self.window)

        # figures
        self.n_fig = n_fig
        self.figs = []
        for idx in range(self.n_fig):
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            fig.figurergba = (1, 1, 1, 0)
            fig.panergba = (1, 1, 1, 0.2)
            self.figs.append(fig)

        # get viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []

        # rgb image to overlay (legacy)
        self.use_rgb_overlay = use_rgb_overlay
        self.loc_rgb_overlay = loc_rgb_overlay

        # rgb images to overlay
        self.rgb_overlay_top_right = None
        self.rgb_overlay_top_left = None
        self.rgb_overlay_bottom_right = None
        self.rgb_overlay_bottom_left = None

        # Perturbation
        self.perturbation = perturbation

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        # g.matid = -1 # newly added (by Jihwan, 2025-02-27)
        """
            mujoco version 3.2 is NOT backward-compatible
        """
        if MUJOCO_VERSION[1] == 1:
            """
            Following lines make error for mujoco version 3.2
            """
            g.texid = -1
            g.texuniform = 0
            g.texrepeat[0] = 1
            g.texrepeat[1] = 1

        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            # setattr(g, key, value)
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                # assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        # Increment number of geoms
        self.scn.ngeom += 1
        return

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(self, camid=None, depth=False):
        if self.render_mode == "window":
            raise NotImplementedError("Use 'render()' in 'window' mode.")

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window
        )
        # update scene
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn,
        )
        # render
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        shape = glfw.get_framebuffer_size(self.window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self.viewport, self.ctx)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            return np.flipud(img)

    def add_overlay(
        self,
        loc="bottom left",
        gridpos=mujoco.mjtGridPos.mjGRID_TOPLEFT,
        text1="",
        text2="",
    ):
        """
        Add overlay
        loc: ['top','top right','top left','bottom','bottom right','bottom left']
        Usage:
            env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOPLEFT,text1='TopLeft')
            env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOP,text1='Top')
            env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,text1='TopRight')
            env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,text1='BottomLeft')
            env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOM,text1='Bottom')
            env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,text1='BottomRight')
        """
        if loc is not None:
            if loc == "top":
                gridpos = mujoco.mjtGridPos.mjGRID_TOP
            elif loc == "top right":
                gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
            elif loc == "top left":
                gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT
            elif loc == "bottom":
                gridpos = mujoco.mjtGridPos.mjGRID_BOTTOM
            elif loc == "bottom right":
                gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
            elif loc == "bottom left":
                gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1
            self._overlay[gridpos][1] += text2
        else:
            self._overlay[gridpos][0] += "\n" + text1
            self._overlay[gridpos][1] += "\n" + text2
        # self._overlay[gridpos][0] += text1 + "\n"
        # self._overlay[gridpos][1] += text2 + "\n"

    def _create_overlay(self):
        """
        Overlay items
        """
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        # self.add_overlay(
        #     gridpos = topleft,
        #     text1   = "A",
        #     text2   = "B",
        # )

    def add_line(
        self,
        fig_idx=0,
        line_idx=0,
        xdata=np.linspace(0, 1, mujoco.mjMAXLINEPNT),
        ydata=np.zeros(mujoco.mjMAXLINEPNT),
        linergb=(0, 0, 1),
        linename="Line Name",
        figurergba=(1, 1, 1, 0),
        panergba=(1, 1, 1, 0.2),
    ):
        """
        Add line to the internal figure
        Usage:
            xdata = np.linspace(start=0.0,stop=10.0,num=100)
            ydata = np.sin(xdata)
            env.viewer.add_line(
                fig_idx=0,line_idx=0,xdata=xdata,ydata=ydata,linergb=(1,0,0),linename='Line 1')
            xdata = np.linspace(start=0.0,stop=10.0,num=100)
            ydata = np.cos(xdata)
            env.viewer.add_line(
                fig_idx=0,line_idx=1,xdata=xdata,ydata=ydata,linergb=(0,0,1),linename='Line 2')
        """
        fig = self.figs[fig_idx]
        fig.figurergba = figurergba
        fig.panergba = panergba
        L = len(xdata)  # this cannot exceed 'mujoco.mjMAXLINEPNT'
        for i in range(L):
            fig.linedata[line_idx][2 * i] = xdata[i]
            fig.linedata[line_idx][2 * i + 1] = ydata[i]
        fig.linergb[line_idx] = linergb
        fig.linename[line_idx] = linename
        fig.linepnt[line_idx] = L

    def add_rgb_overlay(self, rgb_img_raw, fix_ratio=False):
        """
        Set RGB image to render
        """
        width, height = glfw.get_framebuffer_size(self.window)
        rgb_h, rgb_w = height // 4, width // 4
        self.rgb_overlay = np.zeros((rgb_h, rgb_w, 3))
        (h, w) = self.rgb_overlay.shape[:2]
        if fix_ratio:  # fix aspect ratio
            h_raw, w_raw = rgb_img_raw.shape[:2]
            # Calculate scale to preserve aspect ratio
            scale = min(w / w_raw, h / h_raw)
            new_w = int(w_raw * scale)
            new_h = int(h_raw * scale)
            # Resize the image while preserving the aspect ratio
            resized_img = cv2.resize(
                rgb_img_raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
            # Create a black canvas with the target size
            padded_img = np.zeros((h, w, 3), dtype=np.uint8)
            # Calculate the top-left corner for centering the resized image
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2
            # Place the resized image onto the canvas
            padded_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                resized_img
            )
            rgb_img_rsz = padded_img  # Final resized and padded image
        else:
            rgb_img_rsz = cv2.resize(
                rgb_img_raw, (w, h), interpolation=cv2.INTER_NEAREST
            )
        self.rgb_overlay = rgb_img_rsz

    def plot_rgb_overlay(self, rgb=None, loc="top right"):
        """
        loc:['top right','top left','bottom right','bottom left']
        """
        w_window, h_window = glfw.get_framebuffer_size(self.window)
        h_overlay, w_overlay = h_window // 4, w_window // 4
        rgb_overlay = np.zeros((h_overlay, w_overlay, 3))
        # Fix aspect ratio
        h_raw, w_raw = rgb.shape[:2]
        # Calculate scale to preserve aspect ratio
        scale = min(w_overlay / w_raw, h_overlay / h_raw)
        w_new = int(w_raw * scale)
        h_new = int(h_raw * scale)
        # Resize
        rgb_resized = cv2.resize(rgb, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        # Create a black canvas with the target size
        rgb_padded = np.zeros((h_overlay, w_overlay, 3), dtype=np.uint8)
        # Calculate the top-left corner for centering the resized image
        x_offset = (w_overlay - w_new) // 2
        y_offset = (h_overlay - h_new) // 2
        # Place the resized image onto the canvas
        rgb_padded[y_offset : y_offset + h_new, x_offset : x_offset + w_new] = (
            rgb_resized
        )
        # Store the RGB overlay
        if loc == "top right":
            self.rgb_overlay_top_right = rgb_padded
        elif loc == "top left":
            self.rgb_overlay_top_left = rgb_padded
        elif loc == "bottom right":
            self.rgb_overlay_bottom_right = rgb_padded
        elif loc == "bottom left":
            self.rgb_overlay_bottom_left = rgb_padded
        else:
            print(
                "Invalid location for RGB overlay. Use 'top right', 'top left', 'bottom right', or 'bottom left'."
            )

    def reset_rgb_overlay(self, loc=None):
        """
        loc:['top right','top left','bottom right','bottom left']
        """
        if loc is None:
            self.rgb_overlay_top_right = None
            self.rgb_overlay_top_left = None
            self.rgb_overlay_bottom_right = None
            self.rgb_overlay_bottom_left = None
        else:
            if loc == "top_right":
                self.rgb_overlay_top_right = None
            if loc == "top left":
                self.rgb_overlay_top_left = None
            if loc == "bottom right":
                self.rgb_overlay_bottom_right = None
            if loc == "bottom left":
                self.rgb_overlay_bottom_left = None

    def render(self):
        if not self.is_alive:
            raise Exception("GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():

            # Fill overlay items
            self._create_overlay()

            # Render start
            render_start = time.time()
            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)

                # overlay items
                for gridpos, [t1, t2] in self._overlay.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx,
                    )

                # handle figures
                for idx, fig in enumerate(self.figs):
                    width_adjustment = width % 4
                    x = int(3 * width / 4) + width_adjustment
                    y = idx * int(height / 4)
                    viewport = mujoco.MjrRect(x, y, int(width / 4), int(height / 4))
                    # Plot
                    mujoco.mjr_figure(viewport, fig, self.ctx)

                # roverlay rgb images (legacy)
                if self.use_rgb_overlay:
                    rgb_h, rgb_w = height // 4, width // 4
                    if self.loc_rgb_overlay == "top right":
                        left = 3 * rgb_w
                        bottom = 3 * rgb_h
                    elif self.loc_rgb_overlay == "top left":
                        left = 0 * rgb_w
                        bottom = 3 * rgb_h
                    elif self.loc_rgb_overlay == "bottom right":
                        left = 3 * rgb_w
                        bottom = 0 * rgb_h
                    elif self.loc_rgb_overlay == "bottom left":
                        left = 0 * rgb_w
                        bottom = 0 * rgb_h
                    else:
                        print(
                            "Invalid location for RGB overlay. Use 'top right', 'top left', 'bottom right', or 'bottom left'."
                        )
                    self.viewport_rgb_render = mujoco.MjrRect(
                        left=left,
                        bottom=bottom,
                        width=rgb_w,
                        height=rgb_h,
                    )
                    mujoco.mjr_drawPixels(
                        rgb=np.flipud(self.rgb_overlay).flatten(),
                        depth=None,
                        viewport=self.viewport_rgb_render,
                        con=self.ctx,
                    )

                # overlay rgb images
                if self.rgb_overlay_top_right is not None:
                    h_overlay, w_overlay = self.rgb_overlay_top_right.shape[:2]
                    viewport_rgb_top_right = mujoco.MjrRect(
                        left=3 * w_overlay,
                        bottom=3 * h_overlay,
                        width=w_overlay,
                        height=h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb=np.flipud(self.rgb_overlay_top_right).flatten(),
                        depth=None,
                        viewport=viewport_rgb_top_right,
                        con=self.ctx,
                    )
                if self.rgb_overlay_top_left is not None:
                    h_overlay, w_overlay = self.rgb_overlay_top_left.shape[:2]
                    viewport_rgb_top_left = mujoco.MjrRect(
                        left=0 * w_overlay,
                        bottom=3 * h_overlay,
                        width=w_overlay,
                        height=h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb=np.flipud(self.rgb_overlay_top_left).flatten(),
                        depth=None,
                        viewport=viewport_rgb_top_left,
                        con=self.ctx,
                    )
                if self.rgb_overlay_bottom_right is not None:
                    h_overlay, w_overlay = self.rgb_overlay_bottom_right.shape[:2]
                    viewport_rgb_bottom_right = mujoco.MjrRect(
                        left=3 * w_overlay,
                        bottom=0 * h_overlay,
                        width=w_overlay,
                        height=h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb=np.flipud(self.rgb_overlay_bottom_right).flatten(),
                        depth=None,
                        viewport=viewport_rgb_bottom_right,
                        con=self.ctx,
                    )
                if self.rgb_overlay_bottom_left is not None:
                    h_overlay, w_overlay = self.rgb_overlay_bottom_left.shape[:2]
                    viewport_rgb_bottom_left = mujoco.MjrRect(
                        left=0 * w_overlay,
                        bottom=0 * h_overlay,
                        width=w_overlay,
                        height=h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb=np.flipud(self.rgb_overlay_bottom_left).flatten(),
                        depth=None,
                        viewport=viewport_rgb_bottom_left,
                        con=self.ctx,
                    )

                # Double buffering
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

        if self._paused:  # if paused
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

        # clear overlay
        self._overlay.clear()

        # apply perturbation (should this come before mj_step?)
        if self.perturbation:
            self.apply_perturbations()

    def close(self):
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()
