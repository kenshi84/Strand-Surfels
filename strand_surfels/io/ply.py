"""Reading and writing of binary PLY point clouds.

Surfels are stored in the point cloud format of 3D Gaussian Splatting, one
vertex per surfel with the float properties

``x y z nx ny nz f_dc_0..2 f_rest_0..(3K-4) opacity scale_0 scale_1 rot_0..3``

where the SH coefficients are stored channel-major (all coefficients of red,
then green, then blue) and opacities, scales and rotations are raw
(pre-activation) values. The float properties ``nx ny nz`` hold the exported
unit strand tangent (the rotated local x axis). The ``uchar`` properties
``red green blue`` encode that tangent with :func:`fury.colormap.boys2rgb`.
The header comment ``coordinate_frame strands`` identifies exports converted
from the calibrated world coordinates to the original strand coordinates.

Point clouds written since the hair labels were added (a later addition, see
:mod:`strand_surfels.losses.labels`) have one more property, ``label``: the
raw hair label logit (hair probability ``sigmoid(label)``). Readers that do
not know it ignore it; files without it load with ``label`` missing.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Sequence

import numpy as np

STRANDS_FRAME_COMMENT = "coordinate_frame strands"
MILLIMETERS_COMMENT = "length_unit millimeters"

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


def read_ply(
    path: str, element: str = "vertex", *, comments: list[str] | None = None
) -> Dict[str, np.ndarray]:
    """Properties of one element of a binary PLY file, as a dict of 1D arrays.

    Only elements with scalar properties are supported, and the requested
    element must come before any element with list properties. Header comments
    are appended to ``comments`` when supplied.
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
            keyword = words[0] if words else ""
            if keyword == "end_header":
                break
            if keyword == "comment" and comments is not None:
                comments.append(" ".join(words[1:]))
            elif keyword == "format":
                fmt = words[1]
            elif keyword == "element":
                elements.append((words[1], int(words[2]), []))
            elif keyword == "property":
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


def write_ply(
    path: str,
    properties: Dict[str, np.ndarray],
    element: str = "vertex",
    *,
    property_types: dict[str, str] | None = None,
    comments: Sequence[str] = (),
) -> None:
    """Write equal-length 1D arrays with the given PLY scalar types (default: float)."""
    property_types = property_types or {}
    types = {name: property_types.get(name, "float") for name in properties}
    count = len(next(iter(properties.values()))) if properties else 0
    dtype = np.dtype([(name, "<" + _PLY_TYPES[t]) for name, t in types.items()])
    data = np.empty(count, dtype=dtype)
    for name, values in properties.items():
        if len(values) != count:
            raise ValueError("all properties must have the same length")
        data[name] = values
    header = ["ply", "format binary_little_endian 1.0"]
    header.extend(f"comment {comment}" for comment in comments)
    header.append(f"element {element} {count}")
    header += [f"property {t} {name}" for name, t in types.items()]
    header.append("end_header")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def load_surfels_ply(path: str, *, comments: list[str] | None = None) -> Dict[str, np.ndarray]:
    """Raw surfel parameters of a point cloud file (see the module documentation).

    Returns float32 arrays ``xyz [N, 3]``, ``features_dc [N, 1, 3]``,
    ``features_rest [N, K - 1, 3]``, ``opacity [N, 1]``, ``scaling [N, S]``
    and ``rotation [N, 4]``, plus ``label [N, 1]`` if the file has labels.
    """
    v = read_ply(path, comments=comments)

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


def _real_sh_band(degree: int, directions: np.ndarray) -> np.ndarray:
    """Real SH in 3DGS order, omitting the normalization shared by the entire band."""
    # lpmv is a compiled SciPy ufunc.
    # pylint: disable-next=import-outside-toplevel,no-name-in-module
    from scipy.special import factorial, lpmv

    m = np.arange(-degree, degree + 1)
    order = np.abs(m)
    phase = np.arctan2(directions[:, 1:2], directions[:, 0:1]) * order
    scale = np.sqrt(np.where(m == 0, 1, 2) * factorial(degree - order) / factorial(degree + order))
    legendre = lpmv(order, degree, directions[:, 2:3].clip(-1.0, 1.0))
    return scale * legendre * np.where(m < 0, np.sin(phase), np.cos(phase))


def _sh_in_strand_frame(features: np.ndarray, strands_to_world: np.ndarray) -> np.ndarray:
    """Rotate the view-dependent SH coefficients so each camera sees the same colors."""
    rotated = features.copy()
    for degree in range(1, math.isqrt(1 + features.shape[1])):
        # A rotation mixes only the 2l+1 coefficients of one degree. At z=1/2
        # all orders through degree 3 are nonzero, so 2l+1 azimuths span the band.
        count = 2 * degree + 1
        phi = np.arange(count) * (2 * np.pi / count)
        directions = np.column_stack(
            (np.sqrt(0.75) * np.cos(phi), np.sqrt(0.75) * np.sin(phi), np.full(count, 0.5))
        )
        # Y(d_strand) c_strand = Y(d_world) c_world.
        basis = _real_sh_band(degree, directions)
        world_basis = _real_sh_band(degree, directions @ strands_to_world.T)
        change = np.linalg.solve(basis, world_basis).astype(features.dtype)
        band = slice(degree**2 - 1, (degree + 1) ** 2 - 1)  # DC is stored separately
        rotated[:, band] = change @ features[:, band]
    return rotated


def save_surfels_ply(
    path: str,
    params: Dict[str, np.ndarray],
    *,
    strands_to_world_rotation: np.ndarray | None = None,
    output_in_mm: bool = False,
) -> None:
    """Write raw surfels with optional inverse rotation and meter-to-millimeter conversion."""
    # Keep the generic PLY reader and writer usable with NumPy alone.
    # pylint: disable=import-outside-toplevel
    from fury.colormap import boys2rgb
    from scipy.spatial.transform import Rotation

    xyz = np.asarray(params["xyz"], dtype=np.float32)
    n = xyz.shape[0]
    # SH coefficients channel-major: [N, K, 3] -> [N, 3, K] -> [N, 3K]
    f_dc = np.asarray(params["features_dc"]).transpose(0, 2, 1).reshape(n, -1)
    features_rest = np.asarray(params["features_rest"])
    rotation = np.asarray(params["rotation"], dtype=np.float32)
    frame = Rotation.from_quat(rotation[:, [1, 2, 3, 0]])  # PLY wxyz -> SciPy xyzw
    comments = ()
    if strands_to_world_rotation is not None and not np.array_equal(
        strands_to_world_rotation, np.eye(3)
    ):
        strands_to_world = np.asarray(strands_to_world_rotation, dtype=np.float64)
        # Row-vector positions: p_strand = (R.T @ p_world.T).T = p_world @ R.
        xyz = xyz @ strands_to_world.astype(np.float32)
        frame = Rotation.from_matrix(strands_to_world).inv() * frame
        raw_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
        rotation = frame.as_quat()[:, [3, 0, 1, 2]] * raw_norm
        features_rest = _sh_in_strand_frame(features_rest, strands_to_world)
        comments = (STRANDS_FRAME_COMMENT,)
    scaling = np.asarray(params["scaling"])
    if output_in_mm:
        xyz = xyz * 1000.0
        # PLY scale properties store log lengths.
        scaling = scaling + np.log(1000.0)
        comments += (MILLIMETERS_COMMENT,)
    f_rest = features_rest.transpose(0, 2, 1).reshape(n, -1)
    tangents = frame.apply([1.0, 0.0, 0.0]).astype(np.float32)
    props = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    props.update(zip(("nx", "ny", "nz"), tangents.T))
    props.update({f"f_dc_{i}": f_dc[:, i] for i in range(f_dc.shape[1])})
    props.update({f"f_rest_{i}": f_rest[:, i] for i in range(f_rest.shape[1])})
    props["opacity"] = np.asarray(params["opacity"]).reshape(n)
    props.update({f"scale_{i}": scaling[:, i] for i in range(scaling.shape[1])})
    props.update({f"rot_{i}": rotation[:, i] for i in range(rotation.shape[1])})
    if params.get("label") is not None:
        props["label"] = np.asarray(params["label"]).reshape(n)
    color_names = ("red", "green", "blue")
    colors = np.rint(np.clip(boys2rgb(tangents), 0, 1) * 255).astype(np.uint8)
    props.update(zip(color_names, colors.T))
    write_ply(path, props, property_types=dict.fromkeys(color_names, "uchar"), comments=comments)
