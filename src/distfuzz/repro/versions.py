CPU = "https://download.pytorch.org/whl/cpu"
NIGHTLY = "https://download.pytorch.org/whl/nightly/cpu"
NIGHTLY_VERSION = "2.15.0.dev20260926+cpu"

RELEASES = ["2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0", "2.9.1", "2.10.0", "2.11.0", "2.12.1", "2.13.0", "2.14.0"]


def matrix():
    out = []
    for v in RELEASES:
        minor = int(v.split(".")[1])
        out.append(
            dict(
                key=v,
                torch_spec=f"torch=={v}",
                index=CPU,
                numpy="numpy==1.26.4" if minor <= 4 else "numpy==2.2.6",
                pip_extra="",
                nightly=False,
            )
        )
    out.append(
        dict(
            key="nightly",
            torch_spec=f"torch=={NIGHTLY_VERSION}",
            index=NIGHTLY,
            numpy="numpy==2.2.6",
            pip_extra="--pre",
            nightly=True,
        )
    )
    return out


def by_key(key):
    for m in matrix():
        if m["key"] == key:
            return m
    raise KeyError(key)


def order(key):
    """Sort key: releases in numeric order, nightly last."""
    if key == "nightly":
        return (99, 0, 0)
    return tuple(int(x) for x in key.split("."))
