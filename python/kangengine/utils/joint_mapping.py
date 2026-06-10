"""Semantic joint lookup helpers.

Use this when different assets use different bone/body names for the same body
part. Profiles such as GENO, MIXAMO, KW, KW5, and COMMON map semantic labels like
JointSemantic.LEFT_FOOT to asset-specific joint names.

Example:
    mapper = JointMapper.from_motion(motion, profiles=["mixamo", "common"])
    left_foot = mapper.find(JointSemantic.LEFT_FOOT)
    tracking = mapper.find_many([
        JointSemantic.HEAD,
        JointSemantic.LEFT_HAND,
        JointSemantic.RIGHT_HAND,
    ])

If a profile-specific mapping is not found, the mapper falls back through the
remaining profiles in order.
"""

from __future__ import annotations

from enum import Enum


class JointSemantic(str, Enum):
    ROOT = "root"
    HIPS = "hips"
    SPINE = "spine"
    CHEST = "chest"
    NECK = "neck"
    HEAD = "head"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    LEFT_TOE = "left_toe"
    RIGHT_TOE = "right_toe"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"


COMMON: dict[JointSemantic, tuple[str, ...]] = {
    JointSemantic.ROOT: ("root", "hips", "pelvis", "bip01"),
    JointSemantic.HIPS: ("hips", "pelvis", "hip"),
    JointSemantic.SPINE: ("spine", "spine1"),
    JointSemantic.CHEST: ("chest", "spine2", "spine3", "upperchest"),
    JointSemantic.NECK: ("neck", "neck1"),
    JointSemantic.HEAD: ("head", "headtop"),
    JointSemantic.LEFT_SHOULDER: (
        "leftshoulder",
        "left_shoulder",
        "lshoulder",
    ),
    JointSemantic.RIGHT_SHOULDER: (
        "rightshoulder",
        "right_shoulder",
        "rshoulder",
    ),
    JointSemantic.LEFT_ELBOW: ("leftforearm", "leftelbow", "lforearm", "lelbow"),
    JointSemantic.RIGHT_ELBOW: (
        "rightforearm",
        "rightelbow",
        "rforearm",
        "relbow",
    ),
    JointSemantic.LEFT_WRIST: ("leftwrist", "left_hand", "lefthand", "lwrist"),
    JointSemantic.RIGHT_WRIST: (
        "rightwrist",
        "right_hand",
        "righthand",
        "rwrist",
    ),
    JointSemantic.LEFT_HAND: ("lefthand", "left_hand", "lhand", "leftwrist"),
    JointSemantic.RIGHT_HAND: ("righthand", "right_hand", "rhand", "rightwrist"),
    JointSemantic.LEFT_HIP: ("leftupleg", "lefthip", "lhip", "leftthigh"),
    JointSemantic.RIGHT_HIP: ("rightupleg", "righthip", "rhip", "rightthigh"),
    JointSemantic.LEFT_KNEE: ("leftleg", "leftknee", "lknee", "leftshin"),
    JointSemantic.RIGHT_KNEE: ("rightleg", "rightknee", "rknee", "rightshin"),
    JointSemantic.LEFT_ANKLE: ("leftankle", "leftfoot", "lankle"),
    JointSemantic.RIGHT_ANKLE: ("rightankle", "rightfoot", "rankle"),
    JointSemantic.LEFT_FOOT: ("leftfoot", "left_foot", "lfoot", "leftankle"),
    JointSemantic.RIGHT_FOOT: ("rightfoot", "right_foot", "rfoot", "rightankle"),
    JointSemantic.LEFT_TOE: ("lefttoe", "lefttoebase", "ltoe", "leftball"),
    JointSemantic.RIGHT_TOE: ("righttoe", "righttoebase", "rtoe", "rightball"),
    JointSemantic.LEFT_HEEL: ("leftheel", "lheel"),
    JointSemantic.RIGHT_HEEL: ("rightheel", "rheel"),
}

GENO: dict[JointSemantic, tuple[str, ...]] = {
    JointSemantic.ROOT: ("Hips",),
    JointSemantic.HIPS: ("Hips",),
    JointSemantic.SPINE: ("Spine", "Spine1"),
    JointSemantic.CHEST: ("Spine2",),
    JointSemantic.NECK: ("Neck",),
    JointSemantic.HEAD: ("Head",),
    JointSemantic.LEFT_SHOULDER: ("LeftShoulder",),
    JointSemantic.RIGHT_SHOULDER: ("RightShoulder",),
    JointSemantic.LEFT_ELBOW: ("LeftForeArm",),
    JointSemantic.RIGHT_ELBOW: ("RightForeArm",),
    JointSemantic.LEFT_WRIST: ("LeftHand",),
    JointSemantic.RIGHT_WRIST: ("RightHand",),
    JointSemantic.LEFT_HAND: ("LeftHand",),
    JointSemantic.RIGHT_HAND: ("RightHand",),
    JointSemantic.LEFT_HIP: ("LeftUpLeg",),
    JointSemantic.RIGHT_HIP: ("RightUpLeg",),
    JointSemantic.LEFT_KNEE: ("LeftLeg",),
    JointSemantic.RIGHT_KNEE: ("RightLeg",),
    JointSemantic.LEFT_ANKLE: ("LeftFoot",),
    JointSemantic.RIGHT_ANKLE: ("RightFoot",),
    JointSemantic.LEFT_FOOT: ("LeftFoot",),
    JointSemantic.RIGHT_FOOT: ("RightFoot",),
    JointSemantic.LEFT_TOE: ("LeftToe", "LeftToeBase"),
    JointSemantic.RIGHT_TOE: ("RightToe", "RightToeBase"),
}

MIXAMO: dict[JointSemantic, tuple[str, ...]] = {
    JointSemantic.ROOT: ("mixamorig:Hips",),
    JointSemantic.HIPS: ("mixamorig:Hips",),
    JointSemantic.SPINE: ("mixamorig:Spine", "mixamorig:Spine1"),
    JointSemantic.CHEST: ("mixamorig:Spine2",),
    JointSemantic.NECK: ("mixamorig:Neck",),
    JointSemantic.HEAD: ("mixamorig:Head",),
    JointSemantic.LEFT_SHOULDER: ("mixamorig:LeftShoulder",),
    JointSemantic.RIGHT_SHOULDER: ("mixamorig:RightShoulder",),
    JointSemantic.LEFT_ELBOW: ("mixamorig:LeftForeArm",),
    JointSemantic.RIGHT_ELBOW: ("mixamorig:RightForeArm",),
    JointSemantic.LEFT_WRIST: ("mixamorig:LeftHand",),
    JointSemantic.RIGHT_WRIST: ("mixamorig:RightHand",),
    JointSemantic.LEFT_HAND: ("mixamorig:LeftHand",),
    JointSemantic.RIGHT_HAND: ("mixamorig:RightHand",),
    JointSemantic.LEFT_HIP: ("mixamorig:LeftUpLeg",),
    JointSemantic.RIGHT_HIP: ("mixamorig:RightUpLeg",),
    JointSemantic.LEFT_KNEE: ("mixamorig:LeftLeg",),
    JointSemantic.RIGHT_KNEE: ("mixamorig:RightLeg",),
    JointSemantic.LEFT_ANKLE: ("mixamorig:LeftFoot",),
    JointSemantic.RIGHT_ANKLE: ("mixamorig:RightFoot",),
    JointSemantic.LEFT_FOOT: ("mixamorig:LeftFoot",),
    JointSemantic.RIGHT_FOOT: ("mixamorig:RightFoot",),
    JointSemantic.LEFT_TOE: ("mixamorig:LeftToeBase",),
    JointSemantic.RIGHT_TOE: ("mixamorig:RightToeBase",),
}

KW: dict[JointSemantic, tuple[str, ...]] = {
    JointSemantic.ROOT: ("Hips",),
    JointSemantic.HIPS: ("Hips",),
    JointSemantic.CHEST: ("Chest",),
    JointSemantic.NECK: ("Neck",),
    JointSemantic.LEFT_SHOULDER: ("LeftShoulder",),
    JointSemantic.RIGHT_SHOULDER: ("RightShoulder",),
    JointSemantic.LEFT_ELBOW: ("LeftElbow",),
    JointSemantic.RIGHT_ELBOW: ("RightElbow",),
    JointSemantic.LEFT_HIP: ("LeftHip",),
    JointSemantic.RIGHT_HIP: ("RightHip",),
    JointSemantic.LEFT_KNEE: ("LeftKnee",),
    JointSemantic.RIGHT_KNEE: ("RightKnee",),
    JointSemantic.LEFT_ANKLE: ("LeftAnkle",),
    JointSemantic.RIGHT_ANKLE: ("RightAnkle",),
    JointSemantic.LEFT_FOOT: ("LeftAnkle",),
    JointSemantic.RIGHT_FOOT: ("RightAnkle",),
}

KW5: dict[JointSemantic, tuple[str, ...]] = {
    **KW,
    JointSemantic.LEFT_WRIST: ("LeftWrist",),
    JointSemantic.RIGHT_WRIST: ("RightWrist",),
    JointSemantic.LEFT_HAND: ("LeftWrist",),
    JointSemantic.RIGHT_HAND: ("RightWrist",),
    JointSemantic.LEFT_TOE: ("LeftToe",),
    JointSemantic.RIGHT_TOE: ("RightToe",),
}

JOINT_PROFILES: dict[str, dict[JointSemantic, tuple[str, ...]]] = {
    "geno": GENO,
    "mixamo": MIXAMO,
    "kw": KW,
    "kw5": KW5,
    "common": COMMON,
}

DEFAULT_PROFILE_ORDER = ("geno", "mixamo", "kw5", "kw", "common")


def _merged_aliases() -> dict[JointSemantic, tuple[str, ...]]:
    merged: dict[JointSemantic, list[str]] = {}
    for profile_name in DEFAULT_PROFILE_ORDER:
        for semantic, aliases in JOINT_PROFILES[profile_name].items():
            values = merged.setdefault(semantic, [])
            for alias in aliases:
                if alias not in values:
                    values.append(alias)
    return {semantic: tuple(aliases) for semantic, aliases in merged.items()}


JOINT_ALIASES: dict[JointSemantic, tuple[str, ...]] = _merged_aliases()

DEFAULT_TRACKING_SEMANTICS = (
    JointSemantic.HEAD,
    JointSemantic.LEFT_HAND,
    JointSemantic.RIGHT_HAND,
    JointSemantic.LEFT_FOOT,
    JointSemantic.RIGHT_FOOT,
)

DEFAULT_CONTACT_SEMANTICS = (
    JointSemantic.LEFT_FOOT,
    JointSemantic.RIGHT_FOOT,
    JointSemantic.LEFT_TOE,
    JointSemantic.RIGHT_TOE,
    JointSemantic.LEFT_HEEL,
    JointSemantic.RIGHT_HEEL,
)


def normalize_joint_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


class JointMapper:
    def __init__(
        self,
        joint_names: list[str],
        profiles: tuple[str, ...] | list[str] | None = DEFAULT_PROFILE_ORDER,
    ):
        self.joint_names = [str(name) for name in joint_names]
        profiles = DEFAULT_PROFILE_ORDER if profiles is None else profiles
        self.profiles = tuple(str(profile).lower() for profile in profiles)
        self._lowered = [name.lower() for name in self.joint_names]
        self._normalized = [normalize_joint_name(name) for name in self.joint_names]

    @classmethod
    def from_motion(
        cls,
        motion,
        profiles: tuple[str, ...] | list[str] | None = DEFAULT_PROFILE_ORDER,
    ) -> "JointMapper":
        if hasattr(motion, "node_names"):
            return cls([str(name) for name in motion.node_names()], profiles)
        return cls(
            [f"joint_{i}" for i in range(int(motion.num_joints()))],
            profiles,
        )

    def find_name(self, name: str) -> int | None:
        query = str(name).lower()
        query_normalized = normalize_joint_name(query)
        exact_match = next(
            (i for i, candidate in enumerate(self._lowered) if query == candidate),
            None,
        )
        if exact_match is not None:
            return exact_match
        return next(
            (
                i
                for i, candidate in enumerate(self._normalized)
                if query_normalized == candidate
            ),
            None,
        )

    def find_names(self, names: list[str]) -> list[int]:
        indices = []
        for name in names:
            index = self.find_name(name)
            if index is not None and index not in indices:
                indices.append(index)
        return indices

    def aliases(self, semantic: JointSemantic | str) -> tuple[str, ...]:
        semantic = JointSemantic(semantic)
        aliases = []
        for profile_name in self.profiles:
            profile = JOINT_PROFILES.get(profile_name)
            if not profile:
                continue
            for alias in profile.get(semantic, ()):
                if alias not in aliases:
                    aliases.append(alias)
        for alias in JOINT_ALIASES.get(semantic, (semantic.value,)):
            if alias not in aliases:
                aliases.append(alias)
        return tuple(aliases)

    def find(self, semantic: JointSemantic | str) -> int | None:
        for alias in self.aliases(semantic):
            index = self.find_name(alias)
            if index is not None:
                return index
        return None

    def find_many(self, semantics: list[JointSemantic | str]) -> list[int]:
        indices = []
        for semantic in semantics:
            index = self.find(semantic)
            if index is not None and index not in indices:
                indices.append(index)
        return indices

    def default_tracking_indices(self) -> list[int]:
        return self.find_many(list(DEFAULT_TRACKING_SEMANTICS))

    def default_contact_indices(self) -> list[int]:
        return self.find_many(list(DEFAULT_CONTACT_SEMANTICS))
