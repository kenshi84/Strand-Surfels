"""Reading and writing of PLY point clouds (binary, numpy only).

Surfels are stored in the point cloud format of 3D Gaussian Splatting, one
vertex per surfel with the float properties

``x y z nx ny nz f_dc_0..2 f_rest_0..(3K-4) opacity scale_0 scale_1 rot_0..3``

where the SH coefficients are stored channel-major (all coefficients of red,
then green, then blue) and opacities, scales and rotations are raw
(pre-activation) values. The normals are unused and written as zeros.

Point clouds written since the hair labels were added (a later addition, see
:mod:`strand_surfels.losses.labels`) have one more property, ``label``: the
raw hair label logit (hair probability ``sigmoid(label)``). Readers that do
not know it ignore it; files without it load with ``label`` missing.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

# PLY scalar types -> numpy type codes
_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}  # fmt: skip


def read_ply(path: str, element: str = "vertex") -> Dict[str, np.ndarray]:
    """Properties of one element of a binary PLY file, as a dict of 1D arrays.

    Only elements with scalar properties are supported, and the requested
    element must come before any element with list properties.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        elements = []  # [(name, count, [(prop, type) or None for lists])]
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected end of the header")
            words = line.decode("ascii").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                elements.append((words[1], int(words[2]), []))
            elif words[0] == "property":
                is_list = words[1] == "list"
                elements[-1][2].append(None if is_list else (words[2], _PLY_TYPES[words[1]]))
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"{path}: unsupported PLY format {fmt!r} (only binary)")
        endian = "<" if fmt == "binary_little_endian" else ">"

        for name, count, props in elements:
            if any(p is None for p in props):
                raise ValueError(f"{path}: elements with list properties are not supported")
            dtype = np.dtype([(p, endian + t) for p, t in props])
            data = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
            if name == element:
                return {p: data[p].astype(data[p].dtype.newbyteorder("=")) for p, _ in props}
    raise ValueError(f"{path} has no element {element!r}")


def write_ply(path: str, properties: Dict[str, np.ndarray], element: str = "vertex") -> None:
    """Write 1D arrays of equal length as float32 properties of a binary PLY file."""
    names = list(properties)
    count = len(properties[names[0]]) if names else 0
    dtype = np.dtype([(n, "<f4") for n in names])
    data = np.empty(count, dtype=dtype)
    for n in names:
        if len(properties[n]) != count:
            raise ValueError("all properties must have the same length")
        data[n] = properties[n]
    header = ["ply", "format binary_little_endian 1.0", f"element {element} {count}"]
    header += [f"property float {n}" for n in names]
    header.append("end_header")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def load_surfels_ply(path: str) -> Dict[str, np.ndarray]:
    """Raw surfel parameters of a point cloud file (see the module documentation).

    Returns float32 arrays ``xyz [N, 3]``, ``features_dc [N, 1, 3]``,
    ``features_rest [N, K - 1, 3]``, ``opacity [N, 1]``, ``scaling [N, S]``
    and ``rotation [N, 4]``, plus ``label [N, 1]`` if the file has labels.
    """
    v = read_ply(path)

    def stack(prefix: str) -> np.ndarray:
        keys = sorted(
            (k for k in v if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1])
        )
        return np.stack([v[k] for k in keys], axis=1).astype(np.float32)

    n = len(v["x"])
    f_rest = stack("f_rest_") if any(k.startswith("f_rest_") for k in v) else np.zeros((n, 0))
    params = {
        "xyz": np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32),
        "features_dc": stack("f_dc_").reshape(n, 3, 1).transpose(0, 2, 1).copy(),
        "features_rest": f_rest.reshape(n, 3, -1).transpose(0, 2, 1).astype(np.float32).copy(),
        "opacity": v["opacity"].reshape(n, 1).astype(np.float32),
        "scaling": stack("scale_"),
        "rotation": stack("rot_"),
    }
    if "label" in v:
        params["label"] = v["label"].reshape(n, 1).astype(np.float32)
    return params


def save_surfels_ply(path: str, params: Dict[str, np.ndarray]) -> None:
    """Write raw surfel parameters (as returned by :func:`load_surfels_ply`)."""
    xyz = np.asarray(params["xyz"], dtype=np.float32)
    n = xyz.shape[0]
    # SH coefficients channel-major: [N, K, 3] -> [N, 3, K] -> [N, 3K]
    f_dc = np.asarray(params["features_dc"]).transpose(0, 2, 1).reshape(n, -1)
    f_rest = np.asarray(params["features_rest"]).transpose(0, 2, 1).reshape(n, -1)
    props = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    props.update({k: np.zeros(n, dtype=np.float32) for k in ("nx", "ny", "nz")})
    props.update({f"f_dc_{i}": f_dc[:, i] for i in range(f_dc.shape[1])})
    props.update({f"f_rest_{i}": f_rest[:, i] for i in range(f_rest.shape[1])})
    props["opacity"] = np.asarray(params["opacity"]).reshape(n)
    scaling = np.asarray(params["scaling"])
    props.update({f"scale_{i}": scaling[:, i] for i in range(scaling.shape[1])})
    rotation = np.asarray(params["rotation"])
    props.update({f"rot_{i}": rotation[:, i] for i in range(rotation.shape[1])})
    if params.get("label") is not None:
        props["label"] = np.asarray(params["label"]).reshape(n)
    write_ply(path, props)
