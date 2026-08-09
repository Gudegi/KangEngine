# A playback controller for motion editor and physics sim


class PlaybackController:
    def __init__(self):
        self.playing = True
        self._targets = []

    @property
    def has_targets(self) -> bool:
        return bool(self._targets)

    def get_targets(self) -> list:
        return self._targets

    def add_target(self, target):
        if target not in self._targets:
            self._targets.append(target)
        target.set_playing(self.playing)

    def set_playing(self, playing: bool):
        self.playing = bool(playing)
        for target in self._targets:
            target.set_playing(self.playing)

    def toggle_play(self):
        self.set_playing(not self.playing)
        return self.playing
