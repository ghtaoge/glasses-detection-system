from enum import StrEnum


class ClassName(StrEnum):
    NO_GLASSES = "no_glasses"
    EYEGLASSES = "eyeglasses"
    SUNGLASSES = "sunglasses"


CLASS_NAMES = tuple(ClassName)
CLASS_LABELS_ZH = {
    ClassName.NO_GLASSES: "未戴眼镜",
    ClassName.EYEGLASSES: "普通眼镜",
    ClassName.SUNGLASSES: "墨镜",
}


def class_id(name: ClassName) -> int:
    return CLASS_NAMES.index(name)
