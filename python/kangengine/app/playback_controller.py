"""Synchronize motion playback controls and the application's simulation clock."""

import weakref

from .._core import _ke


class PlaybackController:
    def __init__(self, app=None):
        self.playing = True
        self._targets = []
        self._app = weakref.ref(app) if app is not None else None
        self._setting_playing = False

    @property
    def has_targets(self) -> bool:
        return bool(self._targets)

    def get_targets(self) -> list:
        return self._targets

    def add_target(self, target):
        if target in self._targets:
            return
        app = self._app() if self._app is not None else None
        if not self._targets and app is not None:
            self.playing = not app.is_simulation_paused()
        self._targets.append(target)
        target.set_playing(self.playing)
        panel = (
            target if isinstance(target, _ke.MotionSequencerPanel)
            else getattr(target, "panel", None)
        )
        if isinstance(panel, _ke.MotionSequencerPanel):
            # Native callbacks must not keep the controller and its targets alive.
            owner = weakref.ref(self)

            def playing_changed(playing):
                controller = owner()
                if controller is not None:
                    controller.set_playing(playing)

            panel.set_playing_changed_callback(playing_changed)
            if app is not None:
                panel._set_playback_managed(True)
        if app is not None:
            app._set_playback_toggle_callback(self.toggle_play)

    def set_playing(self, playing: bool):
        if self._setting_playing:
            return
        self._setting_playing = True
        try:
            self.playing = bool(playing)
            for target in self._targets:
                target.set_playing(self.playing)
            app = self._app() if self._app is not None else None
            if app is not None:
                app.set_simulation_paused(not self.playing)
        finally:
            self._setting_playing = False

    def toggle_play(self):
        self.set_playing(not self.playing)
        return self.playing
